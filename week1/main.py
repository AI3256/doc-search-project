import pandas as pd
import numpy as np
import sklearn
import os
import sys


# 0. 파일 경로
DATA_PATH = 'data/tech_docs.csv'


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
    result_dict = {}

    for cat in df['category'].unique() :
        cat_text_sums = 0
        cat_counts = cat_counts_df[cat] # 카테고리별 문서 수
        cat_ratio = cat_counts_df[cat] / total_docs * 100 # 카테고리별 비율

        for cat_text in df[df['category'] == cat]['content'] :
            cat_text_sums += len(cat_text.split()) # 특정 카테고리 'content'열의 단어 수 총 합계
        cat_avg_words = cat_text_sums / cat_counts # 특정 카테고리 'content'열의 단어 수 평균

        # 딕셔너리에 저장
        result_dict[cat] = {
            '문서 수' : cat_counts,
            '비율' : cat_ratio,
            '평균 단어 수' : cat_avg_words
        }
    
    print("카테고리\t문서 수\t\t비율\t\t평균 단어 수")
    print("-" * 60)

    for cat, stats in result_dict.items():
        print(f"{cat}\t\t{stats['문서 수']}\t\t{stats['비율']:.2f}%\t\t{stats['평균 단어 수']:.2f}")
    print()

    return result_dict


# 6. main() 함수로 전체 연결
def main() :
    df = load_data(DATA_PATH)
    explore_structure(df)
    show_category_distribution(df)

if __name__ == '__main__' :
    main()