import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import sys
import re


# 0. 데이터 불러오기
def load_data(file_path) :

    if os.path.exists(file_path) :
        df = pd.read_csv(file_path)

    else :
        print(f"오류: '{os.path.basename(file_path)}' 파일을 찾을 수 없습니다.")
        sys.exit()
    
    return df


# 0. 평가셋 불러오기
def load_query(query_path) :
    if os.path.exists(query_path) :
        df = pd.read_csv(query_path)
        print(f'평가셋 크기 : {df.shape[0]}개 질문')
        print()

    else :
        print(f"오류: '{os.path.basename(query_path)}' 파일을 찾을 수 없습니다.")
        sys.exit()
    
    return df


# 1. 전처리 함수
def preprocess(df) :
    df_clean = df.dropna(subset = ['content']).copy()

    def clean_text(text) :
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    df_clean['content_clean'] = df_clean['content'].apply(clean_text)
    
    return df_clean


# 2. 코사인 유사도 직접 구현
def cosine_similarity_numpy(vec1, vec2) :
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0 :
        return 0.0
    
    return dot_product / (norm1 * norm2)


# 3. 키워드 기반 Baseline 검색
def keyword_search(question, df_clean, top_k) :
    # 1. 질문을 소문자로 바꾸고 공백 기준으로 잘라 '단어 집합(set)' 생성
    q_set = set(question.lower().split())

    # 2. 리스트 컴프리헨션을 사용하여 모든 문서에 대해 점수 계산
    score = [len(q_set & set(content.split())) for content in df_clean['content_clean']]
    
    # 3. 점수를 기준으로 상위 K개의 인덱스 추출
    top_indices = np.array(score).argsort()[::-1][:top_k]

    # 4. 추출된 상위 인덱스에 해당하는 데이터 행만 가져오기
    result_score = df_clean.iloc[top_indices].copy()
    
    # 5. 결과 데이터프레임에 해당 문서의 점수(score) 컬럼 추가
    result_score['score'] = np.array(score)[top_indices]

    return result_score


# 4. TF-IDF 벡터화
def build_tfidf(df_clean) :
    # 1. TfidfVectorizer 설정
    vectorizer = TfidfVectorizer(
            max_features = 5000, 
            min_df = 2, 
            stop_words = 'english'
        )
    
    # 2. 모델 학습 및 변환 (fit_transform)
    tfidf_matrix = vectorizer.fit_transform(df_clean['content_clean'])
    
    # 3. 학습된 vectorizer 객체도 함께 반환
    return tfidf_matrix, vectorizer


# 5. TF-IDF 기반 Top-k 검색
def tfidf_search(question, df_clean, tfidf_matrix, vectorizer, top_k) :
    # 1. 질문(Query)을 문서와 동일한 형태의 벡터로 변환
    q_matrix_nparr = vectorizer.transform([question.lower()]).toarray().flatten()
    
    # 2. 문서 집합 전체를 밀집 행렬(Dense Matrix)로 변환
    tfidf_matrix_nparr = tfidf_matrix.toarray()

    # 3. 질문 벡터와 모든 문서 벡터 간의 유사도 계산
    if np.linalg.norm(q_matrix_nparr) == 0:
    # 모든 문서의 유사도를 0으로 설정하여 리스트로 생성
        similarity = [0.0] * tfidf_matrix_nparr.shape[0]
    
    else :
        similarity = [cosine_similarity_numpy(q_matrix_nparr, row) for row in tfidf_matrix_nparr]
    
    # 4. 유사도가 높은 순으로 정렬하여 상위 K개 인덱스 추출
    top_indices = np.array(similarity).argsort()[::-1][:top_k]

    # 5. 선택된 상위 문서 데이터 가져오기
    result_similarity = df_clean.iloc[top_indices].copy()

    # 6. 결과 데이터프레임에 유사도 점수 컬럼 추가
    result_similarity['similarity'] = [similarity[i] for i in top_indices]

    return result_similarity


# 6. Precision at k 구현
def precision_at_k(result_id, truth_id, top_k) :
    '''
    상위 K개 검색 결과 중 실제 정답이 얼마나 포함되어 있는지 계산합니다.
    
    Args:
        result_id (list): 모델이 검색한 문서 ID 리스트
        truth_id (list): 정답 문서 ID 리스트
        top_k (int): 평가할 상위 K개의 범위
        
    Returns:
        float: Precision@K 값 (0.0 ~ 1.0 사이)
    '''
    # 1. 모델이 예측한 결과 중 상위 K개만 슬라이싱합니다.
    # 2. 정답지(truth_id)를 집합(set)으로 변환하여 검색 속도를 최적화합니다.
    # 3. 교집합(&) 연산을 통해 정답과 일치하는 개수를 구합니다.
    # 4. 정밀도는 '전체 검색한 K개 중 정답이 몇 개인가'를 나타냅니다.
    return (len(set(result_id[:top_k]) & set(truth_id))) / top_k


# 7. MRR 구현
def reciprocal_rank(result_id, truth_id) :
    '''
    첫 번째 정답 문서가 몇 번째 순위(rank)에 처음으로 등장하는지 계산합니다.
    순위의 역수를 취하므로 정답이 상위에 있을수록 높은 점수를 얻습니다.
    
    Args:
        result_id (list): 모델이 검색한 문서 ID 리스트
        truth_id (list): 정답 문서 ID 리스트
        
    Returns:
        float: 첫 정답 문서의 순위에 따른 역수 (1/rank), 없으면 0.0
    '''
    # 정답 목록을 set으로 변환하여 검색 속도 최적화
    truth_set = set(truth_id)
    
    # enumerate를 사용하여 순위(1부터 시작)와 해당 문서 ID를 함께 반복합니다.
    # 검색 결과가 정답셋에 포함되어 있다면
    # 해당 순위의 역수를 반환하고 함수를 종료합니다. (가장 먼저 나온 정답만 취급)
    for rank, doc_id in enumerate(result_id, start = 1):
        if doc_id in truth_set:
            return 1 / rank
    
    # 루프가 끝날 때까지 정답을 찾지 못했다면 0점 처리
    return 0.0


# 8. 베이스라인 vs TF-IDF 성능 비교
def run_evaluation(df_query, func, top_k) :
    '''
    주어진 쿼리셋에 대해 모델의 성능(Precision@K, MRR)을 종합적으로 평가합니다.
    
    Args:
        df_query (DataFrame): 평가용 쿼리와 정답 ID가 담긴 데이터프레임
        func (function): 평가할 검색 함수 (wrapper 함수)
        top_k (int): 검색할 상위 K개 수
        
    Returns:
        dict: 전체 쿼리에 대한 평균 Precision과 평균 MRR
    '''
    precisions = []
    mrrs = []
    evaluation_logs = []

    for _, row in df_query.iterrows():
        question = row['query']
        
        # 'D001,D051' 문자열을 리스트 ['D001', 'D051']로 변환
        truth_ids = [x.strip() for x in str(row['relevant_doc_ids']).split(',')]
        
        # 검색 수행
        result_df = func(question, top_k)
        
        # 결과 ID를 리스트로 추출
        # 만약 결과 데이터프레임의 ID도 문자열이면 그대로 사용 가능
        result_ids = [str(x) for x in result_df['doc_id'].tolist()]
        
        # 지표 계산하여 리스트에 쌓음
        precisions.append(precision_at_k(result_ids, truth_ids, top_k))
        mrrs.append(reciprocal_rank(result_ids, truth_ids))
    
        evaluation_logs.append({
            'question': question,
            'truth_ids': row['relevant_doc_ids'],
            'retrieved_ids': ",".join(result_ids)
        })

    # 루프가 끝난 뒤 전체 평균을 계산하여 반환
    results = {
        'precision': np.mean(precisions),
        'mrr': np.mean(mrrs)
    }

    return results, evaluation_logs
    

# 9. 실패 케이스 분석
def analyze_failures(evaluation_logs):
    failures = []

    for log in evaluation_logs:
        # 문자열로 저장된 ID를 다시 리스트로 변환 (필요시)
        truth_ids = [x.strip() for x in str(log['truth_ids']).split(',')]
        retrieved_ids = log['retrieved_ids'].split(',')
        
        # MRR이 0인 경우(상위 K개 안에 정답이 하나도 없는 경우)만 골라냄
        if reciprocal_rank(retrieved_ids, truth_ids) == 0:
            failures.append({
                'query': log['question'],
                'truth': truth_ids,
                'retrieved': retrieved_ids
            })
            
    return failures


# 9. 출력
def display_results(results_key, results_tfidf):
    '''모델별 지표를 비교하여 콘솔에 정렬된 표 형태로 출력합니다.'''
    print(f"=== 성능 비교 ===")
    print(f"{'':<18} {'Precision@3':>12} {'MRR':>8}")
    print(f"{'Keyword Baseline':<18} {results_key['precision']:>12.4f} {results_key['mrr']:>8.4f}")
    print(f"{'TF-IDF':<18} {results_tfidf['precision']:>12.4f} {results_tfidf['mrr']:>8.4f}")
    print()

def print_failures(failures, model_name):
    '''분석된 실패 케이스를 리스트 형식으로 출력합니다.'''
    print(f"=== [{model_name}] 실패 케이스 분석 (총 {len(failures)}개) ===")
    for i, f in enumerate(failures, 1):
        print(f"{i}. Q: {f['query']}")
        print(f"   - 정답 ID: {f['truth']}")
        print(f"   - 검색 결과: {f['retrieved']}")
        print()


# 10. main() 함수로 전체 연결
def main() :
    DATA_PATH = 'data/tech_docs.csv'
    QUERY_PATH = 'data/docs_query.csv'
    QUESTION = 'how does gradient descent work in machine learning'
    TOP_K = 3

    # 0. 데이터 불러오기
    df_raw = load_data(DATA_PATH)

    # 0. 쿼리 불러오기
    df_query = load_query(QUERY_PATH)

    # 1. 전처리
    df_clean = preprocess(df_raw)

    # 4. TF-IDF 벡터화
    tfidf_matrix, vectorizer = build_tfidf(df_clean)

    # 3. 키워드 기반 Baseline 검색
    keyword_search(QUESTION, df_clean, TOP_K)

    # 5. TF-IDF 기반 Top-k 검색
    tfidf_search(QUESTION, df_clean, tfidf_matrix, vectorizer, TOP_K)

    key_wrapper = lambda q, k : keyword_search(q, df_clean, k)
    tfidf_wrapper = lambda q, k : tfidf_search(q, df_clean, tfidf_matrix, vectorizer, k)

    # 7. 베이스라인 vs TF-IDF 성능 비교
    results_keyword, keyword_logs = run_evaluation(df_query, key_wrapper, TOP_K)
    results_tfidf, tfidf_logs = run_evaluation(df_query, tfidf_wrapper, TOP_K)

    # 8. 실패 케이스
    keyword_failures = analyze_failures(keyword_logs)
    tfidf_failures = analyze_failures(tfidf_logs)

    # 9. 출력
    display_results(results_keyword, results_tfidf)
    print_failures(keyword_failures, "Keyword Baseline")
    print_failures(tfidf_failures, "TF-IDF")

    # 10. 결과 csv로 저장
    # pd.DataFrame(key_logs).to_csv("result_keyword.csv", index=False, encoding='utf-8-sig')
    # pd.DataFrame(tfidf_logs).to_csv("result_tfidf.csv", index=False, encoding='utf-8-sig')


if __name__ == '__main__' :
    main()