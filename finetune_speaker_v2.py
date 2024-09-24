import os
import json
import argparse
import itertools
import math
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

import librosa
import logging

# 경고 수준으로 설정하여 Numba 로그의 과도한 출력 방지
logging.getLogger('numba').setLevel(logging.WARNING)

import commons
import utils
from data_utils import (
  TextAudioSpeakerLoader, # 텍스트, 오디오, 스피커 데이터 로더
  TextAudioSpeakerCollate, # 데이터셋 병합
  DistributedBucketSampler # 분산 학습을 위한 버킷 샘플러
)
from models import (
  SynthesizerTrn, # 음성 합성 모델
  MultiPeriodDiscriminator, # 다중 기간 판별기
)
from losses import (
  generator_loss, # 생성기 손실 함수
  discriminator_loss, # 판별기 손실 함수
  feature_loss, # 특징 손실 함수
  kl_loss # KL 다이버전스 손실 함수
)
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch # 멜 스펙트로그램 처리


torch.backends.cudnn.benchmark = True # cuDNN의 성능 벤치마크 사용 설정
global_step = 0 # 전역 학습 단계 변수


def main():
  """Assume Single Node Multi GPUs Training Only"""
  """싱글 노드에서 여러 GPU를 이용한 학습"""
  assert torch.cuda.is_available(), "CPU training is not allowed." "CPU 학습은 지원되지 않습니다."

  n_gpus = torch.cuda.device_count() # 사용 가능한 GPU의 수를 확인
  os.environ['MASTER_ADDR'] = 'localhost' # 분산 학습을 위한 마스터 노드 주소 설정
  os.environ['MASTER_PORT'] = '8000' # 분산 학습을 위한 포트 설정

  hps = utils.get_hparams() # 하이퍼파라미터를 로드
  mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,)) # 여러 GPU를 사용한 학습 실행


def run(rank, n_gpus, hps):
  global global_step
  symbols = hps['symbols'] # 학습에 사용할 심볼 설정
  if rank == 0:
    logger = utils.get_logger(hps.model_dir) # 학습 로그 생성
    logger.info(hps)
    utils.check_git_hash(hps.model_dir) # 모델 디렉토리에 대한 Git 버전 확인
    writer = SummaryWriter(log_dir=hps.model_dir) # TensorBoard 로그 작성
    writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

  # Use gloo backend on Windows for Pytorch
  # PyTorch 분산 프로세스 그룹 초기화 (Windows에서는 gloo, 다른 OS에서는 nccl 사용)
  dist.init_process_group(backend=  'gloo' if os.name == 'nt' else 'nccl', init_method='env://', world_size=n_gpus, rank=rank)
  torch.manual_seed(hps.train.seed) # 랜덤 시드 설정
  torch.cuda.set_device(rank) # 프로세스에 맞는 GPU 설정

  # 데이터셋 로드
  train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps.data, symbols) # 학습 데이터셋 로드
  train_sampler = DistributedBucketSampler( # 분산 학습 샘플러
      train_dataset,
      hps.train.batch_size,
      [32,300,400,500,600,700,800,900,1000],
      num_replicas=n_gpus,
      rank=rank,
      shuffle=True)
  collate_fn = TextAudioSpeakerCollate() # 데이터셋 병합 함수
  train_loader = DataLoader(train_dataset, num_workers=2, shuffle=False, pin_memory=True,
      collate_fn=collate_fn, batch_sampler=train_sampler) # 학습 데이터 로더 생성
  # train_loader = DataLoader(train_dataset, batch_size=hps.train.batch_size, num_workers=2, shuffle=False, pin_memory=True,
  #                           collate_fn=collate_fn)
  if rank == 0:
    eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps.data, symbols) # 검증 데이터셋 로드
    eval_loader = DataLoader(eval_dataset, num_workers=0, shuffle=False,
        batch_size=hps.train.batch_size, pin_memory=True,
        drop_last=False, collate_fn=collate_fn)

  # 모델 생성 및 GPU 할당
  net_g = SynthesizerTrn(
      len(symbols), # 입력으로 사용하는 심볼(문자) 목록의 길이, 즉 모델이 처리할 수 있는 문자 집합의 크기
      hps.data.filter_length // 2 + 1, # 멜 스펙트로그램에서 주파수 축의 크기. 이는 필터 길이의 절반에 1을 더한 값으로, 멜 스펙트로그램의 차원을 설정.
      hps.train.segment_size // hps.data.hop_length, # 세그먼트 크기를 hop 길이로 나눈 값. 이 값은 학습 시 각 오디오 세그먼트의 시간 축의 크기를 결정.
      n_speakers=hps.data.n_speakers, # 스피커의 수, 즉 모델이 처리할 수 있는 스피커(또는 음성)의 수를 설정. 멀티스피커 학습을 위해 사용됨.
      **hps.model).cuda(rank) # 하이퍼파라미터에서 모델 설정을 가져와 네트워크 구조를 정의하고, GPU로 해당 모델을 전송.
  net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)
  # MultiPeriodDiscriminator: 다중 기간 판별기(Discriminator) 모델로, 생성된 오디오가 진짜인지 가짜인지 판별하는 역할을 담당. 주기별로 오디오 데이터를 처리하여 다양한 시간적 특성을 분석.
  # hps.model.use_spectral_norm: 판별기에 스펙트럼 정규화를 적용할지 여부를 설정. 스펙트럼 정규화는 GAN(Generative Adversarial Networks)에서 안정적인 학습을 위해 사용되는 기법.
  # .cuda(rank): 판별기 모델도 GPU로 전송하여 병렬 처리 가능하도록 설정. rank는 GPU 번호를 나타냄.

  # load existing model
  # 기존 모델 체크포인트 로드
  if hps.cont: # 체크포인트에서 학습을 이어서 진행할 경우
      try:
          _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_latest.pth"), net_g, None)
          _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "D_latest.pth"), net_d, None)
          global_step = (epoch_str - 1) * len(train_loader) # 전역 학습 단계 갱신
      except:
          print("Failed to find latest checkpoint, loading G_0.pth...")
          if hps.train_with_pretrained_model:
              print("Train with pretrained model...")
              _, _, _, epoch_str = utils.load_checkpoint("./pretrained_models/G_0.pth", net_g, None)
              _, _, _, epoch_str = utils.load_checkpoint("./pretrained_models/D_0.pth", net_d, None)
          else:
              print("Train without pretrained model...")
          epoch_str = 1
          global_step = 0
  else: # 새로 학습을 시작할 경우
      if hps.train_with_pretrained_model:
          print("Train with pretrained model...")
          _, _, _, epoch_str = utils.load_checkpoint("./pretrained_models/G_0.pth", net_g, None)
          _, _, _, epoch_str = utils.load_checkpoint("./pretrained_models/D_0.pth", net_d, None)
      else:
          print("Train without pretrained model...")
      epoch_str = 1
      global_step = 0

  # freeze all other layers except speaker embedding
  # 스피커 임베딩 레이어를 제외한 다른 레이어의 가중치 고정
  for p in net_g.parameters():
      p.requires_grad = True # 모든 가중치를 학습 가능 상태로 설정
  for p in net_d.parameters():
      p.requires_grad = True # 판별기 가중치도 학습 가능 상태로 설정
  # for p in net_d.parameters():
  #     p.requires_grad = False
  # net_g.emb_g.weight.requires_grad = True

  # 옵티마이저 설정
  optim_g = torch.optim.AdamW(
      net_g.parameters(),
      hps.train.learning_rate,
      betas=hps.train.betas,
      eps=hps.train.eps)
  optim_d = torch.optim.AdamW(
      net_d.parameters(),
      hps.train.learning_rate,
      betas=hps.train.betas,
      eps=hps.train.eps)
  
  # optim_d = None
  # 모델을 DDP(분산 데이터 병렬 처리)로 설정
  net_g = DDP(net_g, device_ids=[rank])
  net_d = DDP(net_d, device_ids=[rank])

  # 학습률 스케줄러 설정
  scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay)
  scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay)

  # 혼합 정밀도 학습을 위한 스케일러 설정
  scaler = GradScaler(enabled=hps.train.fp16_run)

  # 학습 및 평가 루프
  for epoch in range(epoch_str, hps.train.epochs + 1):
    if rank==0: # 메인 프로세스에서만 학습 및 평가
      train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler, [train_loader, eval_loader], logger, [writer, writer_eval])
    else: # 다른 프로세스에서는 학습만 수행
      train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler, [train_loader, None], None, None)
    scheduler_g.step() # 학습률 업데이트
    scheduler_d.step() # 학습률 업데이트


# 학습 및 평가 함수
def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers):
  net_g, net_d = nets
  optim_g, optim_d = optims
  scheduler_g, scheduler_d = schedulers
  train_loader, eval_loader = loaders
  if writers is not None:
    writer, writer_eval = writers

  # train_loader.batch_sampler.set_epoch(epoch)
  global global_step # 전역 학습 단계 변수

  net_g.train()
  net_d.train()

  # 학습 데이터셋에 대해 배치 단위로 학습 수행
  for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, speakers) in enumerate(tqdm(train_loader)):
    x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(rank, non_blocking=True)
    spec, spec_lengths = spec.cuda(rank, non_blocking=True), spec_lengths.cuda(rank, non_blocking=True)
    y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(rank, non_blocking=True)
    speakers = speakers.cuda(rank, non_blocking=True)

    # 혼합 정밀도 학습을 사용하여 forward pass
    with autocast(enabled=hps.train.fp16_run):
      y_hat, l_length, attn, ids_slice, x_mask, z_mask,\
      (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(x, x_lengths, spec, spec_lengths, speakers)

      # 멜 스펙트로그램 생성
      mel = spec_to_mel_torch(
          spec,
          hps.data.filter_length,
          hps.data.n_mel_channels,
          hps.data.sampling_rate,
          hps.data.mel_fmin,
          hps.data.mel_fmax)
      # 멜 스펙트로그램에서 필요한 부분만 자름
      y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length) 
      y_hat_mel = mel_spectrogram_torch(
          y_hat.squeeze(1),
          hps.data.filter_length,
          hps.data.n_mel_channels,
          hps.data.sampling_rate,
          hps.data.hop_length,
          hps.data.win_length,
          hps.data.mel_fmin,
          hps.data.mel_fmax
      )

      # 실제 오디오 데이터를 자름
      y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size) # slice

      # Discriminator
      # 판별기(D)를 사용한 학습
      y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach()) # 판별기 결과 계산
      with autocast(enabled=False):
        loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g) # 판별기 손실 계산
        loss_disc_all = loss_disc # 전체 판별기 손실
    optim_d.zero_grad() # 판별기 옵티마이저의 그래디언트 초기화
    scaler.scale(loss_disc_all).backward() # 손실에 대해 역전파
    scaler.unscale_(optim_d)
    grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None) # 그래디언트 클리핑
    scaler.step(optim_d) # 옵티마이저 업데이트

    # 생성기(G)를 사용한 학습
    with autocast(enabled=hps.train.fp16_run):
      # Generator
      y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat) # 판별기를 다시 사용하여 생성된 오디오 평가
      with autocast(enabled=False):
        loss_dur = torch.sum(l_length.float()) # 지속 시간 손실
        loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel # 멜 스펙트로그램 손실
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl # KL 다이버전스 손실

        loss_fm = feature_loss(fmap_r, fmap_g) # 특징 손실
        loss_gen, losses_gen = generator_loss(y_d_hat_g) # 생성기 손실
        loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl # 전체 생성기 손실

    optim_g.zero_grad() # 생성기 옵티마이저의 그래디언트 초기화
    scaler.scale(loss_gen_all).backward() # 손실에 대해 역전파
    scaler.unscale_(optim_g) 
    grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None) # 그래디언트 클리핑
    scaler.step(optim_g) # 옵티마이저 업데이트
    scaler.update()

    # TensorBoard에 학습 진행 상황을 기록
    if rank==0:
      if global_step % hps.train.log_interval == 0: # 로그 간격에 맞춰 로그 출력
        lr = optim_g.param_groups[0]['lr'] # 현재 학습률 확인
        losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_dur, loss_kl]
        logger.info('Train Epoch: {} [{:.0f}%]'.format(
          epoch,
          100. * batch_idx / len(train_loader)))
        logger.info([x.item() for x in losses] + [global_step, lr])

        scalar_dict = {"loss/g/total": loss_gen_all, "loss/d/total": loss_disc_all, "learning_rate": lr, "grad_norm_g": grad_norm_g}
        scalar_dict.update({"loss/g/fm": loss_fm, "loss/g/mel": loss_mel, "loss/g/dur": loss_dur, "loss/g/kl": loss_kl})

        scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
        scalar_dict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
        scalar_dict.update({"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})
        image_dict = {
            "slice/mel_org": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()), # 실제 멜 스펙트로그램
            "slice/mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()), # 생성된 멜 스펙트로그램
            "all/mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()), # 전체 멜 스펙트로그램
            "all/attn": utils.plot_alignment_to_numpy(attn[0,0].data.cpu().numpy()) # 어텐션 맵
        }
        utils.summarize(
          writer=writer,
          global_step=global_step,
          images=image_dict,
          scalars=scalar_dict)

      # 일정 간격마다 평가 및 체크포인트 저장
      if global_step % hps.train.eval_interval == 0:
        evaluate(hps, net_g, eval_loader, writer_eval)
        
        utils.save_checkpoint(net_g, None, hps.train.learning_rate, epoch,
                              os.path.join(hps.model_dir, "G_latest.pth"))
        
        utils.save_checkpoint(net_d, None, hps.train.learning_rate, epoch,
                              os.path.join(hps.model_dir, "D_latest.pth"))
        # save to google drive
        # 구글 드라이브에 체크포인트 저장
        if os.path.exists("/content/drive/MyDrive/"):
            utils.save_checkpoint(net_g, None, hps.train.learning_rate, epoch,
                                  os.path.join("/content/drive/MyDrive/", "G_latest.pth"))

            utils.save_checkpoint(net_d, None, hps.train.learning_rate, epoch,
                                  os.path.join("/content/drive/MyDrive/", "D_latest.pth"))
            
        # 보존할 체크포인트 수에 따라 오래된 체크포인트 삭제
        if hps.preserved > 0:
          utils.save_checkpoint(net_g, None, hps.train.learning_rate, epoch,
                                  os.path.join(hps.model_dir, "G_{}.pth".format(global_step)))
          utils.save_checkpoint(net_d, None, hps.train.learning_rate, epoch,
                                  os.path.join(hps.model_dir, "D_{}.pth".format(global_step)))
          old_g = utils.oldest_checkpoint_path(hps.model_dir, "G_[0-9]*.pth",
                                               preserved=hps.preserved)  # Preserve 4 (default) historical checkpoints.
          old_d = utils.oldest_checkpoint_path(hps.model_dir, "D_[0-9]*.pth", preserved=hps.preserved)
          if os.path.exists(old_g):
            print(f"remove {old_g}")
            os.remove(old_g)
          if os.path.exists(old_d):
            print(f"remove {old_d}")
            os.remove(old_d)
          if os.path.exists("/content/drive/MyDrive/"):
              utils.save_checkpoint(net_g, None, hps.train.learning_rate, epoch,
                                    os.path.join("/content/drive/MyDrive/", "G_{}.pth".format(global_step)))
              utils.save_checkpoint(net_d, None, hps.train.learning_rate, epoch,
                                    os.path.join("/content/drive/MyDrive/", "D_{}.pth".format(global_step)))
              old_g = utils.oldest_checkpoint_path("/content/drive/MyDrive/", "G_[0-9]*.pth",
                                                   preserved=hps.preserved)  # Preserve 4 (default) historical checkpoints.
              old_d = utils.oldest_checkpoint_path("/content/drive/MyDrive/", "D_[0-9]*.pth", preserved=hps.preserved)
              if os.path.exists(old_g):
                  print(f"remove {old_g}")
                  os.remove(old_g)
              if os.path.exists(old_d):
                  print(f"remove {old_d}")
                  os.remove(old_d)

    global_step += 1 # 전역 학습 단계 업데이트
    if epoch > hps.max_epochs:
        print("Maximum epoch reached, closing training...")
        exit()

  if rank == 0:
    logger.info('====> Epoch: {}'.format(epoch)) # 학습 에포크 기록

# 평가 함수
def evaluate(hps, generator, eval_loader, writer_eval):
    generator.eval() # 모델을 평가 모드로 전환
    with torch.no_grad(): # 평가 시에는 그래디언트를 계산하지 않음
      for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, speakers) in enumerate(eval_loader):
        x, x_lengths = x.cuda(0), x_lengths.cuda(0)
        spec, spec_lengths = spec.cuda(0), spec_lengths.cuda(0)
        y, y_lengths = y.cuda(0), y_lengths.cuda(0)
        speakers = speakers.cuda(0)

        # remove else
        # 1개의 샘플만 추출하여 평가
        x = x[:1]
        x_lengths = x_lengths[:1]
        spec = spec[:1]
        spec_lengths = spec_lengths[:1]
        y = y[:1]
        y_lengths = y_lengths[:1]
        speakers = speakers[:1]
        break

      # 생성기(G)를 사용하여 오디오 생성
      y_hat, attn, mask, *_ = generator.module.infer(x, x_lengths, speakers, max_len=1000)
      y_hat_lengths = mask.sum([1,2]).long() * hps.data.hop_length

      # 멜 스펙트로그램 생성
      mel = spec_to_mel_torch(
        spec,
        hps.data.filter_length,
        hps.data.n_mel_channels,
        hps.data.sampling_rate,
        hps.data.mel_fmin,
        hps.data.mel_fmax)
      y_hat_mel = mel_spectrogram_torch(
        y_hat.squeeze(1).float(),
        hps.data.filter_length,
        hps.data.n_mel_channels,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        hps.data.mel_fmin,
        hps.data.mel_fmax
      )

    # 평가 결과를 이미지와 오디오로 기록
    image_dict = {
      "gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy())
    }
    audio_dict = {
      "gen/audio": y_hat[0,:,:y_hat_lengths[0]]
    }
    if global_step == 0:
      image_dict.update({"gt/mel": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())})
      audio_dict.update({"gt/audio": y[0,:,:y_lengths[0]]})

    utils.summarize(
      writer=writer_eval,
      global_step=global_step,
      images=image_dict,
      audios=audio_dict,
      audio_sampling_rate=hps.data.sampling_rate
    )
    generator.train() # 학습 모드로 복귀


if __name__ == "__main__":
  main() # 메인 함수 실행
