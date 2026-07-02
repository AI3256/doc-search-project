import pandas as pd
import numpy as np
import sklearn
import os
import sys


# 0. 파일 경로
DATA_PATH = 'data/tech_docs copy.csv'


# 1. 데이터 불러오기
def load_data(file_path) :
    print('1. 데이터 불러오기')
    if os.path.exists(file_path) :
        df = pd.read_csv(file_path)
        rows, cols = df.shape
        print(f'데이터 로드 완료: {rows}행 x {cols}열')
    else :
        print(f"오류: '{os.path.basename(file_path)}' 파일을 찾을 수 없습니다.")
        sys.exit()
    print()
    return df


# 2. 데이터 구조 확인
def explore_structure(df) :
    print('2. 데이터 구조 확인')
    print('1) 행/열 수')
    print(f'({df.shape[0]}, {df.shape[1]})')
    print('=' * 70)

    print('2) 컬럼명')
    print(df.columns)
    print('=' * 70)

    print('3) 컬럼별 자료형 목록')
    print(df.dtypes)
    print('=' * 70)

    print('4) 상위 5행')
    print(df.head(5))

    print()


# 3. 카테고리 분포 확인
def show_category_distribution(df) :
    print('3. 카테고리 분포 확인')

    cat_counts_df = df['category'].value_counts()
    total_docs = cat_counts_df.sum() # 전체 문서 수
    result_dict_dist = {}

    for cat in df['category'].unique() :
        cat_text_sums = 0
        cat_counts = cat_counts_df[cat] # 카테고리별 문서 수
        cat_ratio = cat_counts_df[cat] / total_docs * 100 # 카테고리별 비율

        for cat_text in df[df['category'] == cat]['content'] :
            cat_text_sums += len(cat_text.split()) # 특정 카테고리 'content'열의 단어 수 총 합계
        cat_avg_words = cat_text_sums / cat_counts # 특정 카테고리 'content'열의 단어 수 평균

        # 딕셔너리에 저장
        result_dict_dist[cat] = {
            '문서 수' : cat_counts,
            '비율' : cat_ratio,
            '평균 단어 수' : cat_avg_words
        }
    
    print("카테고리\t문서 수\t\t비율\t\t평균 단어 수")
    print("-" * 60)

    for cat, stats in result_dict_dist.items():
        print(f"{cat}\t\t{stats['문서 수']}\t\t{stats['비율']:.2f}%\t\t{stats['평균 단어 수']:.2f}")
    print()

    return result_dict_dist


# 4. 결측치 현황 파악
def check_missing(df) :
    print('4. 결측치 현황 파악')

    result_dict_missing = {}
    missing_cols = []
    clean_cols = []
    total_rows = len(df)

    for col in df.columns :
        missing_count = df[col].isnull().sum()
        if missing_count >= 1 : 
            missing_cols.append(col)
            missing_ratio = missing_count / total_rows * 100

            if missing_ratio < 5 :
                severity = '낮음'
            elif missing_ratio < 20 :
                severity = '주의'
            else :
                severity = '높음'

            result_dict_missing[col] = {
                '결측치 수' : missing_count,
                '결측치 비율' : missing_ratio,
                '심각도' : severity
            }

        else :
            clean_cols.append(col)
            result_dict_missing[col] = {
                '결측치 수' : 0,
                '결측치 비율' : 0,
                '심각도' : '없음'
            }

    if missing_cols :
        print('1) 결측치가 있는 컬럼')
        print("컬럼명\t\t결측치 수\t\t결측치 비율\t\t심각도")
        print("-" * 60)
        # for col, stats in result_dict_missing.items() :
        #     if stats['결측치 수'] >= 1 :
        #         print(f"{col}\t\t{stats['결측치 수']}\t\t{stats['결측치 비율']:.2f}%\t\t{stats['심각도']}")
        #     else :
        #         continue
        for col in missing_cols :
            stats = result_dict_missing[col]
            print(f"{col}\t\t{stats['결측치 수']}\t\t{stats['결측치 비율']:.2f}%\t\t{stats['심각도']}")
        print()

        print('2) 결측치가 없는 컬럼')
        print(', '.join(clean_cols))

    else :
        print('결측치가 있는 컬럼 : 없음')

    return(result_dict_missing)


# 5. 넘파이로 문서 길이 통계량 계산
def numpy_doc_stats(df) :
    print('5. 넘파이로 문서 길이 통계량 계산')
    df_clean = df.dropna().copy() # 어떤 열이든 관계 없이 결측치 있으면 그 행은 제거
    word_counts = np.array([len(str(x).split()) for x in df_clean['content']])

    print('1) content열의 단어 수 통계량(넘파이)')
    np_stats = {
        "평균": np.mean(word_counts),
        "표준편차": np.std(word_counts, ddof = 1),
        "중앙값": np.median(word_counts),
        "최솟값": np.min(word_counts),
        "최댓값": np.max(word_counts)
    }
    
    for k, v in np_stats.items():
        print(f"{k}: {v:.2f}")

    # print(f'평균 : {np.mean(word_counts):.2f}')
    # print(f'표준편차 : {np.std(word_counts, ddof = 1):.2f}')
    # print(f'중앙값 : {np.median(word_counts):.2f}')
    # print(f'최솟값 : {np.min(word_counts):.2f}')
    # print(f'최댓값 : {np.max(word_counts):.2f}')
    print()

    print('2) 50단어 미만 문서')
    short_docs = df_clean[word_counts < 50]
    print(f'50단어 미만 문서 수 : {len(short_docs)}개')

    if len(short_docs) > 0 :
        print('50단어 미만 문서 목록')
        print(short_docs.head())
    print()

    print('3) 넘파이 통계량 vs 판다스 describe() 비교')
    pd_stats = df_clean['content'].apply(lambda x: len(str(x).split())).describe()
    print(f"{'통계량':<10} | {'넘파이':>10} | {'판다스':>10}")
    print("-" * 35)
    print(f"{'평균':<10} | {np_stats['평균']:>10.2f} | {pd_stats['mean']:>10.2f}")
    print(f"{'표준편차':<10} | {np_stats['표준편차']:>10.2f} | {pd_stats['std']:>10.2f}")
    print(f"{'중앙값':<10} | {np_stats['중앙값']:>10.2f} | {pd_stats['50%']:>10.2f}")
    print(f"{'최솟값':<10} | {np_stats['최솟값']:>10.2f} | {pd_stats['min']:>10.2f}")
    print(f"{'최댓값':<10} | {np_stats['최댓값']:>10.2f} | {pd_stats['max']:>10.2f}")
    print()
    
    # 자동 검증
    comparison_items = [('평균', 'mean'), ('표준편차', 'std'), ('중앙값', '50%'), ('최솟값', 'min'), ('최댓값', 'max')]
    
    all_match = True
    for np_key, pd_key in comparison_items:
        if np.isclose(np_stats[np_key], pd_stats[pd_key]):
            print(f"✅ {np_key} 일치")
        else:
            print(f"❌ {np_key} 불일치 (NumPy: {np_stats[np_key]:.2f}, Pandas: {pd_stats[pd_key]:.2f})")
            all_match = False
            
    if all_match:
        print("\n✅ 모든 통계량이 일치합니다.")

# 6. main() 함수로 전체 연결
def main() :
    df = load_data(DATA_PATH)
    # explore_structure(df)
    # show_category_distribution(df)
    # check_missing(df)
    numpy_doc_stats(df)

if __name__ == '__main__' :
    main()