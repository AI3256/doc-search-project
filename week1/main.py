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
    print('=' * 70)

    print()


# 6. main() 함수로 전체 연결
def main() :
    df = load_data(DATA_PATH)
    explore_structure(df)

if __name__ == '__main__' :
    main()