import pandas as pd
import numpy as np
import sklearn
import os
import sys


# 0. 파일 경로
DATA_PATH = 'data/tech_docs.csv'


# 1. 데이터 불러오기
def load_data(file_path) :
    if os.path.exists(file_path) :
        df = pd.read_csv(file_path)
        rows, cols = df.shape
        print(f'데이터 로드 완료: {rows}행 x {cols}열')
    else :
        print(f"오류: '{os.path.basename(file_path)}' 파일을 찾을 수 없습니다.")
        sys.exit()


# 6. main() 함수로 전체 연결
def main() :
    load_data(DATA_PATH)

if __name__ == '__main__' :
    main()