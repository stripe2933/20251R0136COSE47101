import pandas as pd
import numpy as np
import geopandas as gpd
import os
import re
from tqdm import tqdm

path = '../data'
sido = gpd.read_file(os.path.join(path, '../data/ctprvn_20230729/ctprvn.shp'), encoding='cp949')
sgg = gpd.read_file(os.path.join(path, '../data/sig_20230729/sig.shp'), encoding='cp949')
emd = gpd.read_file(os.path.join(path, '../data/emd_20230729/emd.shp'), encoding='cp949')
li = gpd.read_file(os.path.join(path, '../data/li_20230729/li.shp'), encoding='cp949')

sido = sido[['CTP_KOR_NM', 'CTPRVN_CD']]
sgg = sgg[['SIG_KOR_NM', 'SIG_CD']]
emd = emd[['EMD_KOR_NM','EMD_CD']]

fire_df = pd.read_csv(os.path.join(path, '산림청_산불상황관제시스템 산불통계데이터_20241016.csv'), encoding='cp949')

# 공백 제거 함수
def clean_whitespace(text):
    if pd.isna(text):
        return text
    return re.sub(r'\s+', ' ', str(text)).strip()

# 시도명 약식화 함수
def simplify_sido(sido_name):
    if pd.isna(sido_name):
        return sido_name
    
    # 정식 명칭을 약식으로 변환하는 매핑
    sido_mapping = {
        '서울특별시': '서울', 
        '부산광역시': '부산', 
        '대구광역시': '대구',
        '인천광역시': '인천', 
        '광주광역시': '광주', 
        '대전광역시': '대전',
        '울산광역시': '울산', 
        '세종특별자치시': '세종',
        '경기도': '경기', 
        '강원특별자치도': '강원', 
        '강원도': '강원',
        '충청북도': '충북', 
        '충청남도': '충남',
        '전라북도': '전북', 
        '전라남도': '전남',
        '경상북도': '경북', 
        '경상남도': '경남',
        '제주특별자치도': '제주'
    }
    
    return sido_mapping.get(sido_name, sido_name)

# 시군구명 약식화 함수
def simplify_sigungu(sigungu_name):
    if pd.isna(sigungu_name):
        return sigungu_name
    
    # 접미사 제거 패턴
    patterns = [
        (r'(\S+)시\s+(\S+)구$', r'\1 \2'),  # '청주시 흥덕구' -> '청주 흥덕'
        (r'(\S+)시\s+(\S+)군$', r'\1 \2'),  # '창원시 의창군' -> '창원 의창'
        (r'(\S+)시$', r'\1'),            # '수원시' -> '수원'
        (r'(\S+)군$', r'\1'),            # '연천군' -> '연천'
        (r'(\S+)구$', r'\1')             # '강남구' -> '강남'
    ]
    
    # 패턴에 따라 접미사 제거
    for pattern, replacement in patterns:
        match = re.match(pattern, sigungu_name)
        if match:
            if pattern == r'(\S+)시\s+(\S+)구$' or pattern == r'(\S+)시\s+(\S+)군$':
                return f"{match.group(1)} {match.group(2)}"
            else:
                return match.group(1)
    
    return sigungu_name

# 읍면동명 약식화 함수
def simplify_emd(emd_name):
    if pd.isna(emd_name):
        return emd_name
    
    # 접미사 제거 패턴
    patterns = [
        (r'(\S+)읍$', r'\1'),  # '구성읍' -> '구성'
        (r'(\S+)면$', r'\1'),  # '화동면' -> '화동'
        (r'(\S+)동$', r'\1')   # '삼전동' -> '삼전'
    ]
    
    # 패턴에 따라 접미사 제거
    for pattern, replacement in patterns:
        match = re.match(pattern, emd_name)
        if match:
            return match.group(1)
    
    return emd_name

# 공백 제거
for col in ['CTP_KOR_NM', 'SIG_KOR_NM', 'EMD_KOR_NM']:
    if col in sido.columns:
        sido[col] = sido[col].apply(clean_whitespace)
    if col in sgg.columns:
        sgg[col] = sgg[col].apply(clean_whitespace)
    if col in emd.columns:
        emd[col] = emd[col].apply(clean_whitespace)

# 지명 약식화
sido['CTP_simple'] = sido['CTP_KOR_NM'].apply(simplify_sido)
sgg['SIG_simple'] = sgg['SIG_KOR_NM'].apply(simplify_sigungu)
emd['EMD_simple'] = emd['EMD_KOR_NM'].apply(simplify_emd)

# 시도, 시군구, 읍면동 코드 누적 표시
sgg['CTPRVN_CD'] = sgg['SIG_CD'].str[:2]
emd['SIG_CD'] = emd['EMD_CD'].str[:5]
emd['CTPRVN_CD'] = emd['SIG_CD'].str[:2]


# 시도-시군구-읍면, 코드번호 조합 생성

region_list = []

for _, sido_row in tqdm(sido.iterrows(), total=len(sido)):
    ctprvn_cd = sido_row['CTPRVN_CD']
    ctp_simple = sido_row['CTP_simple']
    
    # 해당 시도의 시군구 필터링
    filtered_sgg = sgg[sgg['CTPRVN_CD'] == ctprvn_cd]
    
    for _, sgg_row in filtered_sgg.iterrows():
        sig_cd = sgg_row['SIG_CD']
        sig_simple = sgg_row['SIG_simple']
        
        # 해당 시군구의 읍면동 필터링
        filtered_emd = emd[emd['SIG_CD'] == sig_cd]
        
        for _, emd_row in filtered_emd.iterrows():
            emd_cd = emd_row['EMD_CD']
            emd_simple = emd_row['EMD_simple']
            
            # dictionary 생성
            region_dict = {
                '시도': ctp_simple,
                '시군구': sig_simple,
                '읍면': emd_simple,
                '시도_시군구': f"{ctp_simple} {sig_simple}",
                '시도_시군구_읍면': f"{ctp_simple} {sig_simple} {emd_simple}",
                'CTPRVN_CD': ctprvn_cd,
                'SIG_CD': sig_cd,
                'EMD_CD': emd_cd
            }
            region_list.append(region_dict)

# DataFrame 생성
region_df = pd.DataFrame(region_list)

# 산불 데이터 전처리
fire_df_processed = fire_df.copy()

# 누락 데이터 처리
fire_df_processed['발생장소_시도'] = fire_df_processed['발생장소_시도'].fillna('')
fire_df_processed['발생장소_시군구'] = fire_df_processed['발생장소_시군구'].fillna('')
fire_df_processed['발생장소_읍면'] = fire_df_processed['발생장소_읍면'].fillna('')
fire_df_processed['발생장소_동리'] = fire_df_processed['발생장소_동리'].fillna('')

# 공백 제거
for col in ['발생장소_시도', '발생장소_시군구', '발생장소_읍면', '발생장소_동리']:
    fire_df_processed[col] = fire_df_processed[col].apply(clean_whitespace)

# 시군구 수동 보정
manual_fix_dict = {
    '성남': ['수정', '중원', '분당'],
    '수원': ['장안'],
    '고양': ['덕양', '일산동', '일산서'],
    '부천': ['오정', '원미', '소사'],
    '수원': ['장안', '권선', '팔달', '영통'],
    '안산': ['단원', '상록'],
    '안양': ['동안', '만안'],
    '용인': ['기흥', '수지', '처인'],
    '창원': ['성산', '의창', '마산합포', '진해', '마산회원'],
    '포항': ['남', '북'],
    '천안': ['동남', '서북'],
    '전주': ['완산', '덕진'],
    '청주': ['상당', '서원', '흥덕', '청원'],
}

# 수동 보정 함수 정의
def manual_fix(row):
    시도 = row['발생장소_시도']
    시군구 = row['발생장소_시군구']
    읍면 = row['발생장소_읍면']

    if pd.isna(읍면):
        return row

    if 시군구 in manual_fix_dict:
        if 읍면 in manual_fix_dict[시군구]:
            row['발생장소_시군구'] = f"{시군구} {읍면}"
            row['발생장소_읍면'] = ""
    return row

fire_df_processed = fire_df_processed.apply(manual_fix, axis=1)

# 읍면이 없는 경우 동리 사용
def merge_emd_ri(row):
    if row['발생장소_읍면']:
        return row['발생장소_읍면']
    else:
        return row['발생장소_동리']

fire_df_processed['발생장소_읍면동'] = fire_df_processed.apply(merge_emd_ri, axis=1)

# 군위군 --> 대구광역시 편입 처리
fire_df_processed.loc[
    (fire_df_processed['발생장소_시도'] == '경북') & 
    (fire_df_processed['발생장소_시군구'] == '군위'),
    '발생장소_시도'] = '대구'

# 시도-시군구-읍면 조합 필드 생성
fire_df_processed['발생장소_시도_시군구'] = fire_df_processed.apply(
    lambda row: f"{row['발생장소_시도']} {row['발생장소_시군구']}" if row['발생장소_시도'] and row['발생장소_시군구'] else "", axis=1)
fire_df_processed['발생장소_시도_시군구_읍면동'] = fire_df_processed.apply(
    lambda row: f"{row['발생장소_시도']} {row['발생장소_시군구']} {row['발생장소_읍면동']}" if row['발생장소_시도'] and row['발생장소_시군구'] and row['발생장소_읍면동'] else "", axis=1)

# 산불 데이터 행에 행정구역 코드를 매핑하는 함수
def map_region_codes(row, region_df):
    result = {
        'CTPRVN_CD': None,
        'SIG_CD': None,
        'EMD_CD': None,
        '매핑수준': '매핑 실패',
        '매핑방법': None
    }
    
    # 0. 세종시 특수 처리
    if row['발생장소_시도'] == '세종':
    # 시군구에 읍면이 잘못 들어간 경우 보정
        candidate_match = region_df[
            (region_df['시도'] == '세종') &
            (region_df['읍면'] == row['발생장소_시군구'])
        ]
        if len(candidate_match) == 1:
            # 시군구 --> 읍면 이동, 시군구는 세종으로 고정
            row['발생장소_읍면동'] = row['발생장소_시군구']
            row['발생장소_시군구'] = '세종'

    # 최종적으로 읍면 기준 매칭
        match = region_df[
            (region_df['시도'] == '세종') &
            (region_df['읍면'] == row['발생장소_읍면동'])
        ]
        if len(match) == 1:
            result['CTPRVN_CD'] = match['CTPRVN_CD']
            result['SIG_CD'] = '36110'  # 고정
            result['EMD_CD'] = match['EMD_CD'].values[0]
            result['매핑수준'] = '세종 매칭'
            result['매핑방법'] = '세종 시군구 무시'
            return result


    # 1. 시도-시군구-읍면 완전 매칭
    if row['발생장소_시도_시군구_읍면동']:
        match = region_df[region_df['시도_시군구_읍면'] == row['발생장소_시도_시군구_읍면동']]
        if len(match) == 1:
            result['CTPRVN_CD'] = match['CTPRVN_CD'].values[0]
            result['SIG_CD'] = match['SIG_CD'].values[0]
            result['EMD_CD'] = match['EMD_CD'].values[0]
            result['매핑수준'] = '완전 매칭'
            result['매핑방법'] = '시도,시군구,읍면동'
            return result
    
    # 2. 시도-시군구 매칭 후 읍면 부분 매칭
    if row['발생장소_시도_시군구'] and row['발생장소_읍면동']:
        sig_matches = region_df[region_df['시도_시군구'] == row['발생장소_시도_시군구']]
        if len(sig_matches) > 0:
            # 읍면 부분 매칭 (포함 여부)
            for _, match in sig_matches.iterrows():
                if row['발생장소_읍면동'] in match['읍면'] or match['읍면'] in row['발생장소_읍면동']:
                    result['CTPRVN_CD'] = match['CTPRVN_CD']
                    result['SIG_CD'] = match['SIG_CD']
                    result['EMD_CD'] = match['EMD_CD']
                    result['매핑수준'] = '부분 매칭'
                    result['매핑방법'] = '시도,시군구 완전 / 읍면동 부분'
                    return result
    
    # 3. 시도-시군구만 매칭
    if row['발생장소_시도_시군구']:
        sig_matches = region_df[region_df['시도_시군구'] == row['발생장소_시도_시군구']]
        if len(sig_matches) > 0:
            # 해당 시군구의 첫 번째 읍면동 코드 사용
            result['CTPRVN_CD'] = sig_matches['CTPRVN_CD'].values[0]
            result['SIG_CD'] = sig_matches['SIG_CD'].values[0]
            result['매핑수준'] = '시군구 매칭'
            result['매핑방법'] = '시도,시군구'
            return result
    
    # 4. 시도만 매칭
    if row['발생장소_시도']:
        sido_matches = region_df[region_df['시도'] == row['발생장소_시도']]
        if len(sido_matches) > 0:
            result['CTPRVN_CD'] = sido_matches['CTPRVN_CD'].values[0]
            result['매핑수준'] = '시도 매칭'
            result['매핑방법'] = '시도'
            return result
    
    return result

# 매핑 결과를 저장할 리스트
mapping_results = []

for _, row in tqdm(fire_df_processed.iterrows(), total=len(fire_df_processed)):
    result = map_region_codes(row, region_df)
    mapping_results.append(result)

# 매핑 결과를 DataFrame으로 변환
mapping_df = pd.DataFrame(mapping_results)

# 산불 데이터와 매핑 결과를 병합
fire_df_final = pd.concat([fire_df_processed, mapping_df], axis=1)

# 매핑 결과를 CSV 파일로 저장
fire_df_final.to_csv(os.path.join(path, '산불데이터에코드매핑.csv'), index=False, encoding='cp949')
