import re
from jamo import h2j, j2hcj # 한글 자모를 분리하기 위한 라이브러리
import ko_pron # 한국어 발음을 IPA로 변환하기 위한 라이브러리


# This is a list of Korean classifiers preceded by pure Korean numerals.
# 순수 한국어 숫자와 함께 사용되는 한국어 분류사 목록 (ex: 한 마리, 두 개)
_korean_classifiers = '군데 권 개 그루 닢 대 두 마리 모 모금 뭇 발 발짝 방 번 벌 보루 살 수 술 시 쌈 움큼 정 짝 채 척 첩 축 켤레 톨 통'

# List of (hangul, hangul divided) pairs:
# (한글 음절, 자모로 분리된 한글) 짝 목록:
_hangul_divided = [(re.compile('%s' % x[0]), x[1]) for x in [
    ('ㄳ', 'ㄱㅅ'),
    ('ㄵ', 'ㄴㅈ'),
    ('ㄶ', 'ㄴㅎ'),
    ('ㄺ', 'ㄹㄱ'),
    ('ㄻ', 'ㄹㅁ'),
    ('ㄼ', 'ㄹㅂ'),
    ('ㄽ', 'ㄹㅅ'),
    ('ㄾ', 'ㄹㅌ'),
    ('ㄿ', 'ㄹㅍ'),
    ('ㅀ', 'ㄹㅎ'),
    ('ㅄ', 'ㅂㅅ'),
    ('ㅘ', 'ㅗㅏ'),
    ('ㅙ', 'ㅗㅐ'),
    ('ㅚ', 'ㅗㅣ'),
    ('ㅝ', 'ㅜㅓ'),
    ('ㅞ', 'ㅜㅔ'),
    ('ㅟ', 'ㅜㅣ'),
    ('ㅢ', 'ㅡㅣ'),
    ('ㅑ', 'ㅣㅏ'),
    ('ㅒ', 'ㅣㅐ'),
    ('ㅕ', 'ㅣㅓ'),
    ('ㅖ', 'ㅣㅔ'),
    ('ㅛ', 'ㅣㅗ'),
    ('ㅠ', 'ㅣㅜ')
]]

# List of (Latin alphabet, hangul) pairs:
# (로마자, 한글 발음) 짝 목록:
_latin_to_hangul = [(re.compile('%s' % x[0], re.IGNORECASE), x[1]) for x in [
    ('a', '에이'),
    ('b', '비'),
    ('c', '시'),
    ('d', '디'),
    ('e', '이'),
    ('f', '에프'),
    ('g', '지'),
    ('h', '에이치'),
    ('i', '아이'),
    ('j', '제이'),
    ('k', '케이'),
    ('l', '엘'),
    ('m', '엠'),
    ('n', '엔'),
    ('o', '오'),
    ('p', '피'),
    ('q', '큐'),
    ('r', '아르'),
    ('s', '에스'),
    ('t', '티'),
    ('u', '유'),
    ('v', '브이'),
    ('w', '더블유'),
    ('x', '엑스'),
    ('y', '와이'),
    ('z', '제트')
]]

# List of (ipa, lazy ipa) pairs:
# (IPA 기호, 간략한 IPA 기호) 짝 목록:
_ipa_to_lazy_ipa = [(re.compile('%s' % x[0], re.IGNORECASE), x[1]) for x in [
    ('t͡ɕ','ʧ'), # IPA 't͡ɕ'를 간략화하여 'ʧ'로 변환
    ('d͡ʑ','ʥ'), # IPA 'd͡ʑ'를 간략화하여 'ʥ'로 변환
    ('ɲ','n^'), # IPA 'ɲ'를 간략화하여 'n^'로 변환
    ('ɕ','ʃ'), # IPA 'ɕ'를 'ʃ'로 변환
    ('ʷ','w'), # IPA 'ʷ'를 'w'로 변환
    ('ɭ','l`'),
    ('ʎ','ɾ'),
    ('ɣ','ŋ'),
    ('ɰ','ɯ'),
    ('ʝ','j'),
    ('ʌ','ə'),
    ('ɡ','g'),
    ('\u031a','#'), # Unicode 표기를 간략화하여 '#'로 변환
    ('\u0348','='), # Unicode 표기를 간략화하여 '='로 변환
    ('\u031e',''), # Unicode 표기를 삭제
    ('\u0320',''), # Unicode 표기를 삭제
    ('\u0339','') # Unicode 표기를 삭제
]]

# 라틴 알파벳을 한글 발음으로 변환하는 함수
def latin_to_hangul(text):
    for regex, replacement in _latin_to_hangul:
        text = re.sub(regex, replacement, text)
    return text

# 한글 음절을 자모로 분리하는 함수
def divide_hangul(text):
    text = j2hcj(h2j(text)) # 한글 음절을 자모로 분리
    for regex, replacement in _hangul_divided:
        text = re.sub(regex, replacement, text)
    return text

# 숫자를 한글 발음으로 변환하는 함수 (Sino 또는 Pure 한글 숫자 발음 처리)
def hangul_number(num, sino=True):
    '''Reference https://github.com/Kyubyong/g2pK'''
    num = re.sub(',', '', num) # 숫자에 포함된 쉼표 제거

    if num == '0': # 숫자 '0'은 '영'으로 변환
        return '영'
    if not sino and num == '20': # 순수 한국어 숫자에서 20은 '스무'로 변환
        return '스무'

    digits = '123456789' # 숫자 1~9
    names = '일이삼사오육칠팔구' # 한자 숫자
    digit2name = {d: n for d, n in zip(digits, names)} # 숫자와 한자 숫자의 매핑

    # 순수 한국어 숫자 발음 (한 두 세 네 등) 및 십단위 숫자 발음 (열 스물 등)
    modifiers = '한 두 세 네 다섯 여섯 일곱 여덟 아홉'
    decimals = '열 스물 서른 마흔 쉰 예순 일흔 여든 아흔'
    digit2mod = {d: mod for d, mod in zip(digits, modifiers.split())}
    digit2dec = {d: dec for d, dec in zip(digits, decimals.split())}

    spelledout = []
    for i, digit in enumerate(num): # 각 숫자를 한글 발음으로 변환
        i = len(num) - i - 1 # 자릿수를 계산하여 뒤에서부터 처리
        if sino:
            if i == 0:
                name = digit2name.get(digit, '')
            elif i == 1:
                name = digit2name.get(digit, '') + '십' # 십의 자리는 '십'을 붙임
                name = name.replace('일십', '십') # '일십'은 '십'으로 처리
        else:
            if i == 0:
                name = digit2mod.get(digit, '') # 순수 한국어 숫자 발음 처리
            elif i == 1:
                name = digit2dec.get(digit, '') # 십단위 처리
        if digit == '0': # 숫자가 0일 경우, 특정 조건에서만 공백으로 처리
            if i % 4 == 0:
                last_three = spelledout[-min(3, len(spelledout)):]
                if ''.join(last_three) == '':
                    spelledout.append('')
                    continue
            else:
                spelledout.append('')
                continue
        if i == 2:
            name = digit2name.get(digit, '') + '백' # 백의 자리 처리
            name = name.replace('일백', '백')
        elif i == 3:
            name = digit2name.get(digit, '') + '천' # 천의 자리 처리
            name = name.replace('일천', '천')
        elif i == 4:
            name = digit2name.get(digit, '') + '만' # 만의 자리 처리
            name = name.replace('일만', '만')
        elif i == 5:
            name = digit2name.get(digit, '') + '십'
            name = name.replace('일십', '십')
        elif i == 6:
            name = digit2name.get(digit, '') + '백'
            name = name.replace('일백', '백')
        elif i == 7:
            name = digit2name.get(digit, '') + '천'
            name = name.replace('일천', '천')
        elif i == 8:
            name = digit2name.get(digit, '') + '억' # 억의 자리 처리
        elif i == 9:
            name = digit2name.get(digit, '') + '십'
        elif i == 10:
            name = digit2name.get(digit, '') + '백'
        elif i == 11:
            name = digit2name.get(digit, '') + '천'
        elif i == 12:
            name = digit2name.get(digit, '') + '조' # 조의 자리 처리
        elif i == 13:
            name = digit2name.get(digit, '') + '십'
        elif i == 14:
            name = digit2name.get(digit, '') + '백'
        elif i == 15:
            name = digit2name.get(digit, '') + '천'
        spelledout.append(name) # 변환된 숫자 발음을 결과에 추가
    return ''.join(elem for elem in spelledout)


# 텍스트 내 숫자를 한글 발음으로 변환하는 함수
def number_to_hangul(text):
    '''Reference https://github.com/Kyubyong/g2pK'''
    tokens = set(re.findall(r'(\d[\d,]*)([\uac00-\ud71f]+)', text)) # 숫자와 한글 문자의 조합을 찾음
    for token in tokens:
        num, classifier = token
        if classifier[:2] in _korean_classifiers or classifier[0] in _korean_classifiers:
            spelledout = hangul_number(num, sino=False) # 한국어 분류사가 있을 경우 순수 한국어 숫자로 변환
        else:
            spelledout = hangul_number(num, sino=True) # 그렇지 않으면 한자 숫자로 변환
        text = text.replace(f'{num}{classifier}', f'{spelledout}{classifier}') # 숫자를 한글 발음으로 대체
    # digit by digit for remaining digits
    # 남은 숫자는 자리별로 한글 발음으로 변환
    digits = '0123456789'
    names = '영일이삼사오육칠팔구'
    for d, n in zip(digits, names):
        text = text.replace(d, n)
    return text


# 한국어 텍스트를 Lazy IPA(간략한 IPA)로 변환하는 함수
def korean_to_lazy_ipa(text):
    text = latin_to_hangul(text) # 로마자를 한글 발음으로 변환
    text = number_to_hangul(text) # 숫자를 한글 발음으로 변환
    text=re.sub('[\uac00-\ud7af]+',lambda x:ko_pron.romanise(x.group(0),'ipa').split('] ~ [')[0], text) # 한국어를 IPA로 변환
    for regex, replacement in _ipa_to_lazy_ipa:
        text = re.sub(regex, replacement, text) # IPA 기호를 간략한 IPA 기호로 변환
    return text

# 한국어 텍스트를 IPA로 변환하는 함수
def korean_to_ipa(text):
    text = korean_to_lazy_ipa(text) # Lazy IPA로 변환한 후,
    return text.replace('ʧ','tʃ').replace('ʥ','dʑ') # 특정 기호를 원래 IPA로 변환하여 반환
