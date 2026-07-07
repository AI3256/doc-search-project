import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import sys
import re


# 0. 데이터 불러오기
def load_data(file_path) :
    '''
    지정된 경로에서 CSV 데이터를 불러옵니다.
    
    Args:
        file_path (str): CSV 파일의 경로.
        
    Returns:
        pd.DataFrame: 로드된 데이터프레임.
    '''
    # 파일 경로가 실제 존재하는지 확인하여 불필요한 오류 방지
    if os.path.exists(file_path) :
        df = pd.read_csv(file_path)
        rows, cols = df.shape
        print(f'데이터 로드 완료: {rows}행 x {cols}열')

    else :
        # 파일이 없을 경우 사용자에게 알리고 프로그램 중단
        print(f"오류: '{os.path.basename(file_path)}' 파일을 찾을 수 없습니다.")
        sys.exit()
    
    return df


# 1. 전처리 함수
def preprocess(df) :
    '''
    content 컬럼의 결측치를 제거하고 텍스트를 소문자화 및 특수문자 제거 처리합니다.
    
    Args:
        df (pd.DataFrame): 원본 데이터프레임.
        
    Returns:
        pd.DataFrame: 전처리가 완료된 데이터프레임.
    '''
    df_clean = df.dropna(subset = ['content']).copy()

    def clean_text(text) :
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    df_clean['content_clean'] = df_clean['content'].apply(clean_text)
    print('전처리 완료 : content_clean 컬럼 생성')
    
    return df_clean


# 2. 코사인 유사도 직접 구현
def cosine_similarity_numpy(vec1, vec2) :
    '''
    sklearn 라이브러리 없이 넘파이로 코사인 유사도를 계산합니다.
    
    Args:
        vec1 (np.array): 첫 번째 벡터.
        vec2 (np.array): 두 번째 벡터.
        
    Returns:
        float: 두 벡터 간의 코사인 유사도 (0~1).
    '''
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0 :
        return 0.0
    
    return dot_product / (norm1 * norm2)


# 3. 키워드 기반 Baseline 검색
def keyword_search(question, df_clean, top_k) :
    '''
    질문과 문서 간의 공통 키워드 개수를 계산하여 상위 K개를 검색합니다.
    단어의 빈도(TF)나 중요도(IDF)를 고려하지 않고,
    오직 단어의 출현 여부만 따지는 가장 기본적인 검색 방식
    
    Args:
        question (str): 검색할 질문.
        df_clean (pd.DataFrame): 전처리된 문서 데이터.
        top_k (int): 반환할 상위 문서 개수.
        
    Returns:
        pd.DataFrame: 상위 K개의 문서 결과(score 포함).
    '''
    # 1. 질문을 소문자로 바꾸고 공백 기준으로 잘라 '단어 집합(set)' 생성
    q_set = set(question.lower().split())

    # 2. 리스트 컴프리헨션을 사용하여 모든 문서에 대해 점수 계산
    # 각 문서(content)를 공백 기준으로 잘라 set으로 변환한 뒤, 
    # 질문 단어 집합(q_set)과 교집합(&)을 구하고 그 길이를 측정합니다.
    # 즉, "질문에 있는 단어가 문서에 몇 개나 포함되어 있는가?"를 측정하는 것입니다.
    score = [len(q_set & set(content.split())) for content in df_clean['content_clean']]
    
    # 3. 점수를 기준으로 상위 K개의 인덱스 추출
    # np.array(score).argsort()는 점수가 낮은 순서(오름차순)대로 인덱스를 정렬합니다.
    # [::-1]은 배열을 뒤집어서 점수가 높은 순서(내림차순)로 만듭니다.
    # [:top_k]는 슬라이싱을 통해 상위 K개의 인덱스만 잘라냅니다.
    top_indices = np.array(score).argsort()[::-1][:top_k]

    # 4. 추출된 상위 인덱스에 해당하는 데이터 행만 가져오기
    # iloc은 인덱스 번호를 기반으로 데이터를 선택합니다.
    result_score = df_clean.iloc[top_indices].copy()
    
    # 5. 결과 데이터프레임에 해당 문서의 점수(score) 컬럼 추가
    # 선택된 상위 문서들에 대해서만 점수를 매핑해줍니다.
    result_score['score'] = np.array(score)[top_indices]

    return result_score


# 4. TF-IDF 벡터화
def build_tfidf(df_clean) :
    '''
    TF-IDF 모델을 생성하고 학습하여 벡터 행렬을 반환합니다.
    
    Args:
        df_clean (pd.DataFrame): 전처리된 데이터프레임.
        
    Returns:
        tuple: (tfidf_matrix, vectorizer 객체)
    '''
    # 1. TfidfVectorizer 설정
    # max_features: 빈도수가 높은 상위 5000개의 단어만 사용 (메모리 절약 및 노이즈 제거)
    # min_df=2: 최소 2개 이상의 문서에서 등장한 단어만 사용 (너무 희귀한 단어는 배제)
    # stop_words='english': 'the', 'a', 'is' 등 의미 없는 영어 불용어 자동 제거
    vectorizer = TfidfVectorizer(
            max_features = 5000, 
            min_df = 2, 
            stop_words = 'english'
        )
    
    # 2. 모델 학습 및 변환 (fit_transform)
    # fit: 문서 전체를 읽고 어떤 단어들이 있는지 사전(Vocabulary)을 만듭니다.
    # transform: 각 문서를 단어의 중요도(TF-IDF 점수)를 담은 숫자 행렬로 변환합니다.
    # 0.00의 의미: 해당 문서에 그 단어가 아예 등장하지 않음.
    # 높은 값: 해당 문서에서 그 단어가 매우 중요함 (자주 등장하지만 다른 문서에는 잘 안 나옴).
    # 낮은 값: 해당 문서에 나오긴 하지만 별로 중요하지 않음.
    # 결과값인 tfidf_matrix는 '희소 행렬(Sparse Matrix)' 형태로 메모리 효율이 매우 좋습니다.
    tfidf_matrix = vectorizer.fit_transform(df_clean['content_clean'])

    # 3. 결과 확인
    # rows: 문서의 총 개수, cols: 사전(Vocabulary)에 등록된 단어의 총 개수
    rows, cols = tfidf_matrix.shape
    print(f'TF-IDF 행렬 크기 : ({rows}, {cols}) | 사용된 단어 수 : {cols}')
    
    # 학습된 vectorizer 객체를 함께 반환하는 이유:
    # 나중에 사용자가 질문(Query)을 입력하면, 문서와 똑같은 기준(사전)으로 질문도 벡터화해야 하기 때문입니다.
    return tfidf_matrix, vectorizer


# 5. TF-IDF 기반 Top-k 검색
def tfidf_search(question, df_clean, tfidf_marix, vectorizer, top_k) :
    '''
    질문을 TF-IDF 벡터로 변환 후 코사인 유사도가 높은 상위 K개 문서를 찾습니다.
    
    Args:
        question (str): 검색할 질문.
        df_clean (pd.DataFrame): 전처리된 문서 데이터.
        tfidf_marix (scipy.sparse.csr_matrix): TF-IDF 벡터 행렬.
        vectorizer (TfidfVectorizer): 학습된 벡터라이저.
        top_k (int): 반환할 상위 문서 개수.
        
    Returns:
        pd.DataFrame: 유사도가 계산된 상위 K개의 문서 결과(similarity 포함).
    '''
    # 1. 질문(Query)을 문서와 동일한 형태의 벡터로 변환
    # transform(): 질문에 포함된 단어를 build_tfidf에서 만든 사전(Vocabulary)에 따라 
    #              TF-IDF 점수 벡터로 변환합니다.
    # toarray().flatten(): 희소 행렬을 일반 배열로 바꾸고, 1차원 벡터로 평탄화합니다.
    q_matrix_nparr = vectorizer.transform([question.lower()]).toarray().flatten()

    # 2. 문서 집합 전체를 밀집 행렬(Dense Matrix)로 변환
    # 코사인 유사도 직접 계산을 위해 희소 행렬을 숫자 배열로 변환합니다.
    tfidf_matrix_nparr = tfidf_marix.toarray()

    # 3. 질문 벡터와 모든 문서 벡터 간의 유사도 계산
    # 리스트 컴프리헨션을 사용하여 모든 문서 행(row)에 대해 코사인 유사도를 계산합니다.
    # 결과적으로 [질문1-문서1 유사도, 질문1-문서2 유사도, ...] 형태의 리스트가 생성됩니다.
    similarity = [cosine_similarity_numpy(q_matrix_nparr, row) for row in tfidf_matrix_nparr]
    
    # 4. 유사도가 높은 순으로 정렬하여 상위 K개 인덱스 추출
    top_indices = np.array(similarity).argsort()[::-1][:top_k]

    # 5. 선택된 상위 문서 데이터 가져오기
    # iloc을 사용하여 결과에 해당하는 행만 데이터프레임으로 추출합니다.
    result_similarity = df_clean.iloc[top_indices].copy()

    # 6. 결과 데이터프레임에 유사도 점수 컬럼 추가
    # 리스트 인덱싱을 통해 추출된 문서의 유사도 점수만 매핑합니다.    
    result_similarity['similarity'] = [similarity[i] for i in top_indices]

    return result_similarity


# 6. 출력
def display_results(question, baseline_df, tfidf_df):
    '''결과를 포맷에 맞춰 콘솔에 출력합니다.'''

    display_df = tfidf_df.copy()
    display_df['similarity'] = display_df['similarity'].map('{:.4f}'.format)

    print(f'\n질문 : {question}\n')
    
    print('=== Keyword Baseline ===')
    print(baseline_df[['doc_id', 'title', 'category', 'score']])
    print()
    
    print('=== TF-IDF Search ===')
    print(display_df[['doc_id', 'title', 'category', 'similarity']])


# 7. main() 함수로 전체 연결
def main() :
    DATA_PATH = 'data/tech_docs.csv'
    QUESTION = 'how does gradient descent work in machine learning'
    TOP_K = 3

    # 0. 데이터 불러오기
    df_raw = load_data(DATA_PATH)

    # 1. 전처리
    df_clean = preprocess(df_raw)

    # 4. TF-IDF 벡터화
    tfidf_matrix, vectorizer = build_tfidf(df_clean)

    # 3. 키워드 기반 Baseline 검색
    result_score = keyword_search(QUESTION, df_clean, TOP_K)

    # 5. TF-IDF 기반 Top-k 검색
    result_similarity = tfidf_search(QUESTION, df_clean, tfidf_matrix, vectorizer, TOP_K)

    # 6. 출력
    display_results(QUESTION, result_score, result_similarity)

if __name__ == '__main__' :
    main()