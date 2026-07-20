import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import os
import sys
import re


# 0. 데이터 불러오기
def load_data(file_path) :
    '''
    지정된 경로에서 CSV 파일을 읽어 데이터프레임으로 반환합니다.
    
    Args :
        file_path(str) : 로드할 데이터 파일의 경로
        
    Returns :
        df(pd.DataFrame): 로드된 데이터프레임 객체
        
    Raises :
        SystemExit : 파일이 존재하지 않을 경우 오류 메시지를 출력하고 프로그램을 종료함.
    '''
    if os.path.exists(file_path) :
        df = pd.read_csv(file_path)
        rows, cols = df.shape
        print(f'데이터 로드 완료: {rows}행 x {cols}열')
        print()
    else :
        print(f"오류: '{os.path.basename(file_path)}' 파일을 찾을 수 없습니다.")
        sys.exit()
    
    return df


# 0. 평가셋 불러오기
def load_query(query_path) :
    '''
    평가용 쿼리 데이터가 담긴 CSV 파일을 불러옵니다.

    Args:
        query_path (str): 쿼리 데이터가 저장된 CSV 파일의 경로.

    Returns:
        pd.DataFrame: 불러온 쿼리 데이터가 담긴 데이터프레임.

    Raises:
        SystemExit: 파일이 존재하지 않을 경우 오류 메시지를 출력하고 프로그램을 종료함.
    '''
    if os.path.exists(query_path) :
        df = pd.read_csv(query_path)

    else :
        print(f"오류: '{os.path.basename(query_path)}' 파일을 찾을 수 없습니다.")
        sys.exit()
    
    return df


# 1-1. 텍스트 정제
def clean_text(text) :
    '''
    입력된 텍스트를 소문자로 변환하고, 영문자와 숫자, 공백을 제외한 특수문자를 제거합니다.
    
    Args:
        text (str) : 정제할 원본 텍스트.
        
    Returns:
        str : 특수문자가 제거되고 소문자로 변환된 텍스트.
    '''
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
    

# 1-2. 기본 전처리 함수
def preprocess(df, target_col = 'content', new_col = 'content_clean') :
    '''
    content 컬럼의 결측치를 제거하고, 지정된 컬럼의 텍스트를 정제하여 새로운 컬럼에 저장합니다.
    
    Args:
        df (pd.DataFrame) : 원본 데이터프레임.
        target_col (str) : 전처리할 대상 컬럼명.
        new_col (str) : 전처리된 텍스트를 저장할 새로운 컬럼명.
        
    Returns:
        pd.DataFrame : 결측치가 제거되고 전처리된 컬럼이 추가된 데이터프레임.
    '''
    df_clean = df.dropna(subset = [target_col]).copy()
    df_clean[new_col] = df_clean[target_col].apply(clean_text)
    return df_clean


# 1-3. 제목을 3회 반복해 본문 앞에 붙인 전처리 함수
def preprocess_improved(df_clean, target_col = 'content', new_col = 'content_clean') :
    '''
    제목(title)을 3회 반복하여 본문(content) 앞에 붙여 새로운 전처리 컬럼을 생성합니다.
    
    Args:
        df_clean (pd.DataFrame) : 기본 전처리가 완료된 데이터프레임 (title 컬럼 포함).
        target_col (str) : 반복할 본문 텍스트가 담긴 컬럼명.
        new_col (str) : 결과를 저장할 새로운 컬럼명.
        
    Returns:
        pd.DataFrame : 제목이 강조된 텍스트 컬럼이 추가된 데이터프레임.
    '''
    df_imp = df_clean.copy()
    title_clean = df_imp['title'].apply(clean_text)
    title_weighted = (title_clean + " ") * 3
    df_imp[new_col] = title_weighted + df_imp[target_col]
    return df_imp


# 2. 키워드 기반 Baseline 검색
def keyword_search(question, df_clean, top_k, target_col='content_clean') :
    '''
    질문과 문서 간의 공통 키워드 개수를 계산하여 상위 K개를 검색합니다.
    단어의 중요도(TF-IDF 등)를 고려하지 않고 단어 출현 여부만 따집니다.
    
    Args:
        question (str) : 검색할 질문 문자열.
        df_clean (pd.DataFrame) : 전처리된 문서 데이터프레임.
        top_k (int) : 반환할 상위 문서 개수.
        target_col (str) : 검색 대상이 되는 텍스트 컬럼명.
        
    Returns:
        pd.DataFrame : 검색 결과(score 컬럼 포함)가 포함된 상위 K개의 문서 데이터프레임.
    '''
    # 1. 질문을 소문자로 바꾸고 공백 기준으로 잘라 '단어 집합(set)' 생성
    q_set = set(question.lower().split())

    # 2. 리스트 컴프리헨션을 사용하여 모든 문서에 대해 점수 계산
    score = [len(q_set & set(content.split())) for content in df_clean[target_col]]
    
    # 3. 점수를 기준으로 상위 K개의 인덱스 추출
    top_indices = np.array(score).argsort()[::-1][:top_k]

    # 4. 추출된 상위 인덱스에 해당하는 데이터 행만 가져오기
    result_score = df_clean.iloc[top_indices].copy()
    
    # 5. 결과 데이터프레임에 해당 문서의 점수(score) 컬럼 추가
    result_score['score'] = np.array(score)[top_indices]

    return result_score


# 3. TF-IDF 벡터화
def build_tfidf(df_clean, target_col = 'content_clean', label = 'Base') :
    '''
    TF-IDF 모델을 생성하고 학습하여 정규화된 벡터 행렬을 반환합니다.
    
    Args:
        df_clean (pd.DataFrame) : 전처리된 데이터프레임.
        target_col (str) : 벡터화할 텍스트가 담긴 컬럼명.
        label (str) : 출력 로그에 표시할 모델 레이블.
        
    Returns:
        tuple : (scipy.sparse.csr_matrix, TfidfVectorizer) 정규화된 행렬과 학습된 벡터라이저 객체.
    '''
    # 1. TfidfVectorizer 설정
    vectorizer = TfidfVectorizer(
            max_features = 5000, 
            min_df = 2, 
            stop_words = 'english'
        )
    
    # 2. 모델 학습 및 변환 (fit_transform)
    tfidf_matrix = vectorizer.fit_transform(df_clean[target_col])
    
    # 2. 정규화된 행렬 생성 (이게 속도 향상의 핵심!)
    tfidf_matrix_norm = normalize(tfidf_matrix, axis=1)

    rows, cols = tfidf_matrix.shape
    print(f'[{label}] TF-IDF 행렬 크기 : ({rows}, {cols}) | 사용된 단어 수 : {cols}')
    print()

    # 3. 학습된 vectorizer 객체도 함께 반환
    return tfidf_matrix_norm, vectorizer


# 4. TF-IDF 기반 Top-k 검색
def tfidf_search(question, df_clean, tfidf_matrix_norm, vectorizer, top_k) :
    '''
    질문을 TF-IDF 벡터로 변환 후, 사전 계산된 행렬과 코사인 유사도가 높은 상위 K개 문서를 찾습니다.
    
    Args:
        question (str) : 검색할 질문 문자열.
        df_clean (pd.DataFrame) : 문서 데이터프레임.
        tfidf_matrix_norm (scipy.sparse.csr_matrix) : 미리 계산된 정규화된 TF-IDF 문서 행렬.
        vectorizer (TfidfVectorizer) : 학습된 TF-IDF 벡터라이저 객체.
        top_k (int) : 반환할 상위 문서 개수.
        
    Returns:
        pd.DataFrame : 유사도(similarity) 점수가 포함된 상위 K개의 문서 데이터프레임.
    '''
    # 1. 질문(Query)을 문서와 동일한 형태의 벡터로 변환
    q_vec = vectorizer.transform([question.lower()])
    q_vec_norm = normalize(q_vec, axis=1)

    # 2. 희소 행렬 상태에서 바로 코사인 유사도 계산
    # 결과값은 (1, 60) 형태의 2차원 배열이 됨
    similarities = (q_vec_norm @ tfidf_matrix_norm.T).toarray().flatten()
    
    # 3. 정렬 및 추출
    top_indices = similarities.argsort()[::-1][:top_k]

    result_similarity = df_clean.iloc[top_indices].copy()
    result_similarity['similarity'] = similarities[top_indices]

    return result_similarity


# 5. tfidf_search 함수가 제대로 동작하는지 질문 1개로 확인
def test_tfidfsearch(question, tfidf_df):
    '''
    TF-IDF 검색 결과를 화면에 출력하여 동작 여부를 확인합니다.

    Args:
        question (str): 검색에 사용된 질문(쿼리) 문자열.
        tfidf_df (pd.DataFrame): 검색 결과가 담긴 데이터프레임. 
                                 'similarity' 컬럼이 포함되어 있어야 합니다.
    
    Returns:
        None: 결과를 콘솔에 출력하기만 합니다.
    '''
    testresult = tfidf_df.copy()
    testresult['similarity'] = testresult['similarity'].map('{:.4f}'.format)

    print(f'=== 예시 검색 : {question} ===')
    print(testresult[['doc_id', 'title', 'category', 'similarity']])
    print()


# 6. Precision at k 구현
def precision_at_k(result_id, truth_id, top_k) :
    '''
    상위 K개 검색 결과 중 실제 정답(relevant) 문서가 포함된 비율(정밀도)을 계산합니다.
    
    Args:
        result_id (list) : 모델이 검색한 문서 ID 리스트.
        truth_id (list) : 실제 정답 문서 ID 리스트.
        top_k (int) : 평가할 상위 문서 개수.
        
    Returns:
        float : 계산된 Precision@K 값 (0.0 ~ 1.0).
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
        tuple : (dict: 지표 평균값, list: 상세 평가 로그)
    '''
    precisions = []
    mrrs = []
    evaluation_logs = []

    for row in df_query.itertuples(index=False):
        question = row.query
        
        # 'D001,D051' 문자열을 리스트 ['D001', 'D051']로 변환
        truth_ids = [x.strip() for x in str(row.relevant_doc_ids).split(',')]
        
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
            'truth_ids': truth_ids,
            'retrieved_ids': result_ids
        })

    # 루프가 끝난 뒤 전체 평균을 계산하여 반환
    results = {
        'precision': np.mean(precisions),
        'mrr': np.mean(mrrs)
    }

    return {'precision': np.mean(precisions), 'mrr': np.mean(mrrs)}, evaluation_logs
    

# 9. 실패 케이스 분석
def analyze_failures(evaluation_logs):
    '''
    검색 결과 상위 K개 내에 정답이 하나도 포함되지 않은(정밀도 0) 실패 케이스를 추출합니다.
    
    Args:
        evaluation_logs (list) : run_evaluation에서 생성된 로그 리스트.
        
    Returns:
        list : 실패한 쿼리에 대한 상세 로그 리스트.
    '''
    failures = []
    for log in evaluation_logs:
        if not (set(log['retrieved_ids']) & set(log['truth_ids'])):
            failures.append(log)
    return failures


# 10-1. 성능 출력
def print_summary_table(title, data_list):
    '''데이터 리스트를 받아 요약 테이블을 콘솔에 출력합니다.'''
    df = pd.DataFrame(data_list)
    df = df.rename(columns={'precision': 'Precision@3', 'mrr': 'MRR'})
    
    print(f"=== {title} ===")
    print(df)
    print()

def print_all_results(results):
    '''전체 모델의 성능 비교 표를 출력합니다.'''
    # 1. 첫 번째 표: 키워드 vs TF-IDF
    print_summary_table("키워드/TF-IDF 성능 비교", [
        {'Model': 'Keyword Baseline', **results['Keyword_Base']},
        {'Model': 'TF-IDF',           **results['TFIDF_Base']}
    ])

    # 2. 두 번째 표: 전처리 비교
    print_summary_table("전처리(기본/제목 반복) 성능 비교", [
        {'Algorithm': 'Keyword', 'Method': 'Base',     **results['Keyword_Base']},
        {'Algorithm': 'Keyword', 'Method': 'Weighted', **results['Keyword_Imp']},
        {'Algorithm': 'TF-IDF',  'Method': 'Base',     **results['TFIDF_Base']},
        {'Algorithm': 'TF-IDF',  'Method': 'Weighted', **results['TFIDF_Imp']}
    ])


# 10-2. 실패 케이스 출력
def print_failures(failures, model_name):
    '''실패 케이스의 상세 내용을 리스트 형식으로 출력합니다.'''
    print(f"=== [{model_name}] 실패 케이스 분석 (총 {len(failures)}개) ===")
    for i, f in enumerate(failures, 1):
        print(f"{i}. Q: {f['question']}")
        print(f"   - 정답 ID: {f['truth_ids']}")
        print(f"   - 검색 결과: {f['retrieved_ids']}")
        print()


# 11. main() 함수로 전체 연결
def main() :
    DATA_PATH = 'data/tech_docs.csv'
    QUERY_PATH = 'data/docs_query.csv'
    QUESTION = 'git merge conflicts'
    TOP_K = 3

    # 0. 데이터 불러오기
    df_raw = load_data(DATA_PATH)

    # 0. 쿼리 불러오기
    df_query = load_query(QUERY_PATH)

    # 1. 전처리 기본 / 제목 강화 2가지 생성
    df_clean = preprocess(df_raw)
    df_imp = preprocess_improved(df_clean, target_col = 'content_clean', new_col = 'content_weighted')
    
    # 2. TF-IDF 벡터화 (기본 / 제목 강화 2가지)
    tfidf_matrix, vectorizer = build_tfidf(df_clean, label = 'Base')
    tfidf_matrix_imp, vectorizer_imp = build_tfidf(df_imp, target_col = 'content_weighted', label = 'Weighted')

    # 3. TF-IDF 기반 Top-k 검색
    result_similarity = tfidf_search(QUESTION, df_clean, tfidf_matrix, vectorizer, TOP_K)

    # 4. 예시 질문 하나로 tfidf_search 실행해서 검색이 동작하는지 확인
    test_tfidfsearch(QUESTION, result_similarity)

    # 5. 검색 모델 설정
    models = {
        'Keyword_Base':   lambda q, k: keyword_search(q, df_clean, k, target_col='content_clean'),
        'Keyword_Imp':    lambda q, k: keyword_search(q, df_imp, k, target_col='content_weighted'),
        'TFIDF_Base':     lambda q, k: tfidf_search(q, df_clean, tfidf_matrix, vectorizer, k),
        'TFIDF_Imp':      lambda q, k: tfidf_search(q, df_imp, tfidf_matrix_imp, vectorizer_imp, k)
    }

    # 6. 평가 및 결과 저장
    results = {}
    logs = {}

    for name, func in models.items():
        results[name], logs[name] = run_evaluation(df_query, func, TOP_K)

    # 7. 결과 출력
    print_all_results(results)
    
    # 8. 실패 케이스 출력
    print_failures(analyze_failures(logs['TFIDF_Base']), "TF-IDF")


if __name__ == '__main__' :
    main()