import pandas as pd
import numpy as np
import sklearn
import os
import sys
import re

# 설정값
MISSING_THRESHOLD_LOW = 5
MISSING_THRESHOLD_HIGH = 20
WORD_COUNT_THRESHOLD = 50

# 1. 데이터 불러오기
def load_data(file_path) :
    '''
    지정된 경로에서 CSV 파일을 읽어 데이터프레임으로 반환합니다.
    
    Args :
        file_path(str) : 분석할 데이터 파일의 경로
        
    Returns :
        df(pd.DataFrame): 로드된 데이터프레임 객체
        
    Raises :
        SystemExit : 파일이 존재하지 않을 경우 프로그램을 안전하게 종료함
    '''
    print('1. 데이터 불러오기')

    # 파일 경로가 실제 존재하는지 확인하여 불필요한 오류 방지
    if os.path.exists(file_path) :
        df = pd.read_csv(file_path)
        rows, cols = df.shape
        print(f'데이터 로드 완료: {rows}행 x {cols}열')
    else :
        # 파일이 없을 경우 사용자에게 알리고 프로그램 중단
        print(f"오류: '{os.path.basename(file_path)}' 파일을 찾을 수 없습니다.")
        sys.exit()
    
    print('\n')
    
    return df


# 1-1. 데이터 전처리
def preprocess_data(df):
    '''
    데이터셋에서 결측치를 제거하고, 분석에 적합한 문자열 데이터만 필터링합니다.

    Args :
        df_raw(pd.DataFrame) : 로드된 원본 데이터프레임

    Returns:
        df_clean(pd.DataFrame): 결측치와 비문자열(non-string) content가 제거된 정제된 데이터프레임
    '''
    # # 1. 모든 컬럼에서 결측치(NaN)가 포함된 행 제거
    # 만약 content 열만 타겟팅해야 한다면 df_clean = df.dropna(subset=['content']).copy()로 변경 가능합니다.
    df_clean = df.dropna().copy()

    # 2. 'content' 컬럼의 데이터가 문자열(str)인 행만 추출
    # (문자열이 아닌 데이터로 인한 오류 방지)
    df_clean = df_clean[df_clean['content'].apply(lambda x: isinstance(x, str))]
    
    return df_clean


# 2. 데이터 구조 확인
def explore_structure(df) :
    '''
    데이터프레임의 기본적인 구조와 정보를 요약하여 출력합니다.
    
    Args :
        df(pd.DataFrame) : 분석할 원본 데이터프레임
    
    Returns :
        None : 별도의 반환값 없이 콘솔에 결과를 출력합니다.
    '''
    print('2. 데이터 구조 확인')

    # 1) 행과 열의 개수를 출력하여 전체 데이터 규모 파악
    print('1) 행/열 수')
    print(f'({df.shape[0]}, {df.shape[1]})')
    print('=' * 70)

    # 2) 데이터 내 컬럼명 확인
    print('2) 컬럼명')
    print(df.columns)
    print('=' * 70)

    # 3) 각 컬럼의 데이터 타입(dtype) 확인하여 데이터 형식이 적절한지 검토
    print('3) 컬럼별 자료형 목록')
    print(df.dtypes)
    print('=' * 70)

    # 4) 데이터의 첫 5행을 출력하여 실제 데이터 내용 확인
    print('4) 상위 5행')
    print(df.head(5))

    print('\n')


# 3. 카테고리 분포 확인
def show_category_distribution(df_clean) :
    '''
    카테고리별 문서 수, 비율, 평균 단어 수를 계산하고 표 형태로 출력합니다.

    Args : 
        df(pd.DataFrame) : 분석할 원본 데이터프레임

    Returns :
        dict : 카테고리명을 키로 하고, 통계량(문서 수, 비율, 평균 단어 수)을 값으로 가지는 딕셔너리
    '''
    
    print('3. 카테고리 분포 확인')

    cat_counts_df = df_clean['category'].value_counts()
    total_docs = cat_counts_df.sum() # 전체 문서 수
    result_dict_dist = {}

    # 각 카테고리별로 순회하며 통계량을 계산합니다.
    for cat in df_clean['category'].unique() :
        cat_text_sums = 0
        cat_counts = cat_counts_df[cat] # 카테고리별 문서 수
        cat_ratio = cat_counts_df[cat] / total_docs * 100 # 카테고리별 비율

        # 해당 카테고리의 모든 문서 단어 수를 합산합니다.
        for cat_text in df_clean[df_clean['category'] == cat]['content'] :
            cat_text_sums += len(cat_text.split())
        
        # 해당 카테고리의 평균 단어 수를 계산합니다.
        cat_avg_words = cat_text_sums / cat_counts

        # 결과값을 딕셔너리에 저장합니다.
        result_dict_dist[cat] = {
            '문서 수' : cat_counts,
            '비율' : cat_ratio,
            '평균 단어 수' : cat_avg_words
        }
    
    # 결과를 표로 출력합니다.
    print("카테고리\t문서 수\t\t비율\t\t평균 단어 수")
    print("-" * 60)

    for cat, stats in result_dict_dist.items():
        print(f"{cat}\t\t{stats['문서 수']}\t\t{stats['비율']:.2f}%\t\t{stats['평균 단어 수']:.2f}")
    
    print('\n')

    return result_dict_dist


# 4. 결측치 현황 파악
def check_missing(df) :
    '''
    데이터프레임의 각 컬럼별 결측치 수와 비율을 계산하고, 
    비율에 따라 심각도(낮음/주의/높음)를 분류하여 출력합니다.

    Args :
        df(pd.DataFrame) : 분석할 원본 데이터프레임

    Returns :
        dict : 컬럼별 결측치 통계(수, 비율, 심각도)를 담은 딕셔너리
    '''
    print('4. 결측치 현황 파악')

    result_dict_missing = {}
    missing_cols = []
    clean_cols = []
    total_rows = len(df)

    for col in df.columns :
        missing_count = df[col].isnull().sum()

        # 결측치가 하나라도 존재하는 경우
        if missing_count >= 1 : 
            missing_cols.append(col)
            missing_ratio = missing_count / total_rows * 100

            # 결측치 비율에 따른 심각도 분류
            if missing_ratio < MISSING_THRESHOLD_LOW :
                severity = '낮음'
            elif missing_ratio < MISSING_THRESHOLD_HIGH :
                severity = '주의'
            else :
                severity = '높음'

            result_dict_missing[col] = {
                '결측치 수' : missing_count,
                '결측치 비율' : missing_ratio,
                '심각도' : severity
            }
        
        # 결측치가 없는 경우
        else :
            clean_cols.append(col)
            result_dict_missing[col] = {
                '결측치 수' : 0,
                '결측치 비율' : 0,
                '심각도' : '없음'
            }

    # 결측치 정보 출력
    if missing_cols :
        print('1) 결측치가 있는 컬럼')
        print(f"{'컬럼명':<10} | {'결측치 수':<10} | {'결측치 비율':<10} | {'심각도':<10}")
        print("-" * 60)

        for col in missing_cols :
            stats = result_dict_missing[col]
            print(f"{col:<10} | {stats['결측치 수']:>10} | {stats['결측치 비율']:>10.2f}% | {stats['심각도']:<10}")
        print()

        print('2) 결측치가 없는 컬럼')
        print(', '.join(clean_cols))
        print('\n')

    else :
        print('결측치가 있는 컬럼 : 없음')
        print('\n')

    return(result_dict_missing)


# 5. 넘파이로 문서 길이 통계량 계산
def numpy_doc_stats(df_clean) :
    '''
    NumPy를 사용해 문서 길이의 통계량을 계산하고, Pandas의 결과와 일치하는지 비교합니다.

    Args :
        df(pd.DataFrame) : 분석할 원본 데이터프레임

    Returns :
        None : 분석 결과를 콘솔에 출력합니다.
    '''
    print('5. 넘파이로 문서 길이 통계량 계산')
    
    
    # 문자열을 분리하여 단어 수를 배열로 변환
    word_counts = np.array([len(str(x).split()) for x in df_clean['content']])

    print('1) content열의 단어 수 통계량(넘파이)')
    np_stats = {
        '평균': np.mean(word_counts),
        '표준편차': np.std(word_counts, ddof = 1), # ddof=1: 표본 표준편차 설정
        '중앙값': np.median(word_counts),
        '최솟값': np.min(word_counts),
        '최댓값': np.max(word_counts)
    }
    
    for k, v in np_stats.items():
        print(f"{k}: {v:.2f}")
    print()

    # 50단어 미만인 문서만 필터링하여 확인
    print('2) 50단어 미만 문서')
    short_docs = df_clean[word_counts < WORD_COUNT_THRESHOLD]
    print(f'50단어 미만 문서 수 : {len(short_docs)}개')

    if len(short_docs) > 0 :
        print('50단어 미만 문서 목록')
        print(short_docs.head())
    print()

    # Pandas의 describe() 함수 결과와 넘파이 결과 비교
    print('3) 넘파이 통계량 vs 판다스 describe() 비교')
    pd_stats = df_clean['content'].apply(lambda x: len(str(x).split())).describe()
    
    print(f"{'통계량':<10} | {'넘파이':>10} | {'판다스':>10}")
    print("-" * 50)

    stats_list = [
        ('평균', np_stats['평균'], pd_stats['mean']),
        ('표준편차', np_stats['표준편차'], pd_stats['std']),
        ('중앙값', np_stats['중앙값'], pd_stats['50%']),
        ('최솟값', np_stats['최솟값'], pd_stats['min']),
        ('최댓값', np_stats['최댓값'], pd_stats['max'])
    ]

    for name, np_val, pd_val in stats_list:
        print(f'{name:<12} | {np_val:>10.2f} | {pd_val:>10.2f}')
    print()

    # 자동 검증 : 부동 소수점 오차를 고려하여 일치 여부 확인
    comparison_items = [('평균', 'mean'), ('표준편차', 'std'), ('중앙값', '50%'), ('최솟값', 'min'), ('최댓값', 'max')]
    
    all_match = True
    for np_key, pd_key in comparison_items:
        if np.isclose(np_stats[np_key], pd_stats[pd_key]):
            print(f'✅ {np_key} 일치')
        else:
            print(f'❌ {np_key} 불일치 (NumPy: {np_stats[np_key]:.2f}, Pandas: {pd_stats[pd_key]:.2f})')
            all_match = False
            
    if all_match:
        print('\n✅ 모든 통계량 일치')




# 1. 전처리 함수
def preprocess(df) :
    df_clean = df.dropna(subset = ['content']).copy()

    def clean_text(text) :
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    df_clean['content_clean'] = df_clean['content'].apply(clean_text)

    print(df_clean[['content', 'content_clean']].head(3))
    
    return df_clean
    # for x in df_clean['content_clean'] :
    #     print(x)


# 2. 코사인 유사도 직접 구현
def consine_similarity_numpy(vec1, vec2) :
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0 :
        return 0.0
    
    return np.dot(vec1, vec2) / (norm1 * norm2)


# 3. 키워드 기반 Baseline 검색
def keyword_search(question, df_clean, top_k) :
    q_set = set(question.lower().split())

    df_clean['score'] = df_clean['content_clean'].apply(lambda x : len(q_set & set(x.split())))
    result_score = df_clean.sort_values('score', ascending = False)
    print(result_score[['doc_id', 'title', 'category', 'score']].head(top_k))

    return result_score



# 6. main() 함수로 전체 연결
def main() :
    '''
    프로그램의 전체 실행 흐름을 제어합니다.
    데이터 로드부터 구조 확인, 결측치 조사, 통계 분석까지의 과정을 순차적으로 실행합니다.
    '''
    DATA_PATH = 'data/tech_docs.csv'

    # 1. 데이터 불러오기
    df_raw = load_data(DATA_PATH)

    # # 1-1. 데이터 정제하기
    # df_clean = preprocess_data(df_raw)

    # # 2. 데이터 구조 확인
    # explore_structure(df_raw)

    # # 3. 카테고리 분포 확인
    # show_category_distribution(df_clean)

    # # 4. 결측치 현황 파악
    # check_missing(df_raw)

    # # 5. 넘파이를 활용한 문서 길이 통계량 계산 및 비교 검증
    # numpy_doc_stats(df_clean)


    QUESTION = 'how does gradient descent work in machine learning'
    TOP_K = 3

    # 1. 전처리
    df_clean = preprocess(df_raw)

    # 2. 코사인 유사도 직접 구현
    # consine_similarity_numpy(v1, v2)

    # 3. 키워드 기반 Baseline 검색
    keyword_search(QUESTION, df_clean, TOP_K)


if __name__ == '__main__' :
    main()