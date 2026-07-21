# 🔍 Tech Docs Search Engine & Evaluation Pipeline

텍스트 전처리 방식과 검색 알고리즘(Keyword Baseline vs TF-IDF)에 따른 정보 검색(Information Retrieval) 성능을 평가하고 비교하기 위한 파이썬 파이프라인 프로젝트입니다.

---

## 📂 주요 기능 (Features)

1. **유연한 쿼리 및 데이터 로드**
   - 기술 문서 데이터셋(CSV)과 평가 쿼리셋을 안정적으로 로드합니다.
   - **일회성/단발성 테스트 지원**: CSV 파일 경로뿐만 아니라, 파이썬 리스트 형태의 쿼리를 즉시 입력하여 평가셋으로 변환할 수 있습니다 (`load_query`).
2. **다양한 전처리 파이프라인**
   - **기본 전처리 (`preprocess`)**: 소문자 변환, 특수문자 제거 및 결측치 처리.
   - **강화 전처리 (`preprocess_improved`)**: 문서의 중요 키워드인 제목(`title`)을 3회 반복하여 본문 앞에 가중치로 부여하는 방식 구현.
3. **검색 알고리즘 비교**
   - **Keyword Baseline**: 단순 단어 출현 빈도 및 교집합 기반 검색 (`keyword_search`).
   - **TF-IDF + Cosine Similarity**: 사이킷런 기반 TF-IDF 벡터화 및 정규화된 희소 행렬 연산을 활용한 고속 코사인 유사도 검색 (`tfidf_search`).
4. **자동화된 랜덤 샘플 테스트**
   - 평가 쿼리셋(`df_query`)에서 무작위로 질문을 추출하여 실시간으로 검색 결과를 콘솔에 시각화하고 정상 동작 여부를 검증합니다 (`test_tfidfsearch`).
5. **정량적 성능 평가 및 실패 케이스 분석**
   - **Precision@K** 및 **MRR (Mean Reciprocal Rank)** 지표를 활용한 객관적 성능 측정 (`run_evaluation`).
   - 모델이 상위 K개 내에 정답을 찾지 못한 **실패 케이스(Failure Cases)**를 자동으로 추출하여 분석 (`analyze_failures`).
6. **가독성 높은 결과 출력**
   - 콘솔 출력 시 데이터 정렬(문자열 좌측, 지표 우측 정렬)과 소수점 포맷팅(`{:.4f}`)이 적용된 요약 비교 테이블 제공 (`print_summary_table`).

---

## 🛠️ 기술 스택 (Tech Stack)

- **Language**: Python 3.x
- **Libraries**: 
  - `pandas`, `numpy` (데이터 조작 및 연산)
  - `scikit-learn` (`TfidfVectorizer`, `cosine_similarity`, `normalize`)
  - `re`, `os`, `sys` (텍스트 정제 및 시스템 제어)

---

## 🚀 시작하기 (Getting Started)

### 1. 사전 준비 (Prerequisites)
필수 라이브러리를 설치합니다.
```bash
pip install pandas numpy scikit-learn
```

### 2. 프로젝트 구조
```
project_root/
│
├── data/
│   ├── tech_docs.csv        # 검색 대상 기술 문서 데이터
│   └── docs_query.csv       # 평가용 쿼리 및 정답 ID셋 (선택)
│
├── main.py                  # 메인 실행 스크립트 (파이프라인)
└── README.md
```

### 3. 코드 실행
스크립트 하단의 main() 함수에서 QUERY_INPUT 설정을 통해 CSV 파일 경로 혹은 테스트용 리스트 중 원하는 방식을 선택하여 실행할 수 있습니다.

💡 사용 방법 예시 (QUERY_INPUT)

A. CSV 파일 기반 전체 평가
```python
QUERY_INPUT = 'data/docs_query.csv'
```

B. 즉석 리스트 쿼리 기반 일회성/단발성 테스트
```python
QUERY_INPUT = [
    {'query': 'git merge conflicts', 'relevant_doc_ids': 'D018, D014'},
    {'query': 'how to undo last commit', 'relevant_doc_ids': 'D016, D054'}
]
```