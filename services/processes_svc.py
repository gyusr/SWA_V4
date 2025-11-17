from schemas import processes_schemas
from typing import List, Dict, Tuple, Optional

from newspaper import Article
import asyncio
import httpx

from google import genai
from dotenv import load_dotenv
import os

# [추가] S-BERT 관련 임포트
from sentence_transformers import SentenceTransformer, util
import torch # S-BERT는 내부적으로 PyTorch 사용

# Celery(prefork)와 PyTorch 충돌 방지 (GPU 워커용)
try:
    if torch.multiprocessing.get_start_method() != 'spawn':
        torch.multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

load_dotenv()
GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")
CUSTOM_SEARCH_JSON_API_KEY=os.getenv("CUSTOM_SEARCH_JSON_API_KEY")
CUSTOM_SEARCH_ENGINE_API_KEY=os.getenv("CUSTOM_SEARCH_ENGINE_API_KEY")

# 상수 정의
MAX_NEWS_SEARCH = 50
NEWS_PER_REQUEST = 10
TOP_N_NEWS = 5

# [수정] 전역 변수는 선언만 해두고, 실제 로드는 하지 않음 (None)
sbert_model = None

# [신규] 모델을 필요할 때만 로드하는 함수 (Singleton 패턴)
def get_sbert_model():
    global sbert_model
    if sbert_model is None:
        print("S-BERT 모델 로드 시작 (Lazy Loading)...")
        try:
            # 모델 로드
            sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            print("S-BERT 모델 로드 완료.")
        except Exception as e:
            print(f"S-BERT 모델 로드 실패: {e}")
            return None
    return sbert_model


async def summary_meeting_and_keyword_meeting_by_gemini(
    original_meeting: str
):
    # ... (기존 코드 동일) ...
    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = f"회의록: {original_meeting}"
    response = await client.aio.models.generate_content(
        model='gemini-2.0-flash',
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": processes_schemas.SummaryMeetingAndKeywordMeetingByLLM.model_json_schema(),
        },
    )
    summary_meeting_and_keyword_meeting = processes_schemas.SummaryMeetingAndKeywordMeetingByLLM.model_validate_json(response.text)
    summary_meeting = summary_meeting_and_keyword_meeting.summary_meeting
    keyword_meeting_list = summary_meeting_and_keyword_meeting.keyword_meeting_list
    return summary_meeting, keyword_meeting_list


async def news_url_list_by_custom_search_json_api(
    keyword_meeting_list: List[str]
) -> List[str]:
    # ... (기존 코드 동일) ...
    if not CUSTOM_SEARCH_JSON_API_KEY or not CUSTOM_SEARCH_ENGINE_API_KEY:
        return []

    query = " ".join(keyword_meeting_list) + " news -filetype:pdf"
    API_URL = "https://www.googleapis.com/customsearch/v1"
    
    num_requests = MAX_NEWS_SEARCH // NEWS_PER_REQUEST
    start_indices = [1 + i * NEWS_PER_REQUEST for i in range(num_requests)]

    tasks = []
    timeout_config = httpx.Timeout(30.0) 
    
    async with httpx.AsyncClient(timeout=timeout_config) as client:
        for start_index in start_indices:
            params = {
                "key": CUSTOM_SEARCH_JSON_API_KEY,
                "cx": CUSTOM_SEARCH_ENGINE_API_KEY,
                "q": query,
                "num": NEWS_PER_REQUEST,
                "start": start_index,
                "sort": "date"
            }
            tasks.append(client.get(API_URL, params=params))

        print(f"DEBUG: Google Custom Search 병렬 요청 {len(tasks)}개 시작")
        try:
            responses = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            print(f"DEBUG: asyncio.gather 실행 중 오류: {e}")
            return []

    all_news_urls = set()
    for idx, res in enumerate(responses):
        if isinstance(res, Exception):
            continue
        if res.status_code != 200:
            continue
        try:
            results = res.json()
            urls = [item['link'] for item in results.get('items', [])]
            for url in urls:
                all_news_urls.add(url) 
        except Exception as e:
            pass

    return list(all_news_urls)


def _crawl_one_article(url: str) -> Tuple[Optional[str], Optional[str]]:
    # ... (기존 코드 동일) ...
    try:
        article = Article(url)
        article.download()
        article.parse()
        if article.title and article.text:
            return article.title, article.text
        else:
            return None, None
    except Exception as e:
        return None, None

async def original_news_list_by_newspaper3k(
    news_url_list: List[str]
) -> List[Dict[str, Optional[str]]]:
    # ... (기존 코드 동일) ...
    if not news_url_list:
        return []
    print(f"DEBUG: Newspaper3k 크롤링 시작 (총 {len(news_url_list)}개 URL)")
    tasks = []
    for url in news_url_list:
        tasks.append(asyncio.to_thread(_crawl_one_article, url))
    results = await asyncio.gather(*tasks)
    zipped_results = zip(news_url_list, results)
    news_items_list = []
    
    for url, (title, text) in zipped_results:
        if title and text:
            news_item = {
                "url": url,
                "title": title,
                "original": text,
                "summary": None
            }
            news_items_list.append(news_item)
    print(f"DEBUG: 크롤링 성공 (총 {len(news_items_list)}개)")
    return news_items_list


# [수정] S-BERT 관련도 분석 함수 (내부에서 get_sbert_model 호출)
def _run_sbert_similarity(
    summary_meeting: str, 
    news_items_list: List[Dict[str, Optional[str]]]
) -> List[Dict[str, Optional[str]]]:
    """
    [동기 함수] S-BERT를 사용하여 유사도 계산.
    이 함수가 호출될 때만 모델을 로드합니다.
    """
    
    # [수정] 여기서 모델을 가져옵니다. (CPU 워커는 이 함수를 호출하지 않으므로 모델을 로드하지 않음)
    model = get_sbert_model()
    
    if not model:
        print("  [S-BERT] 오류: S-BERT 모델 로드 실패. 상위 5개 반환.")
        return news_items_list[:TOP_N_NEWS]

    if not news_items_list:
        return []
        
    print(f"  [S-BERT] S-BERT 인코딩 시작 (회의록 1개, 뉴스 {len(news_items_list)}개)")
    
    try:
        # 1. 회의록 요약본 임베딩
        meeting_embedding = model.encode(
            summary_meeting, 
            convert_to_tensor=True
        )
        
        # 2. 뉴스 본문(original) 리스트 임베딩
        corpus_texts = [
            item['original'] for item in news_items_list if item.get('original')
        ]
        valid_news_items = [
            item for item in news_items_list if item.get('original')
        ]
        
        if not valid_news_items:
            return []

        corpus_embeddings = model.encode(
            corpus_texts, 
            convert_to_tensor=True
        )
        
        # 3. 코사인 유사도 계산
        cosine_scores = util.cos_sim(meeting_embedding, corpus_embeddings)[0]
        
        # 4. 상위 Top-N 선별
        scores_with_indices = list(zip(cosine_scores, range(len(valid_news_items))))
        sorted_scores = sorted(scores_with_indices, key=lambda x: x[0], reverse=True)
        top_n_indices = [idx for score, idx in sorted_scores[:TOP_N_NEWS]]
        top_n_news_items = [valid_news_items[i] for i in top_n_indices]
        
        print(f"  [S-BERT] Top {len(top_n_news_items)}개 선별 완료.")
        return top_n_news_items
        
    except Exception as e:
        print(f"  [S-BERT] 오류 발생: {e}")
        return news_items_list[:TOP_N_NEWS]


# [삭제] async def cosine_similarity_by_sbert wrapper는 더 이상 사용하지 않음 (Celery Task가 직접 동기함수 호출)
# (아래 summary_news 함수는 유지)

async def summary_news_list_by_gemini(
    news_items_list_selected: List[Dict[str, Optional[str]]]
) -> List[Dict[str, Optional[str]]]:
    # ... (기존 코드 동일) ...
    client = genai.Client(api_key=GEMINI_API_KEY)
    print(f"DEBUG: Gemini 뉴스 요약 시작 (총 {len(news_items_list_selected)}개)")

    for i in range(len(news_items_list_selected)):
        if not news_items_list_selected[i].get('original'):
            continue

        prompt = f"다음 뉴스를 한국어로 3~4문장으로 요약해줘: {news_items_list_selected[i]['original']}"
        try:
            response = await client.aio.models.generate_content(
                model='gemini-2.0-flash',
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": processes_schemas.SummaryNewsByLLM.model_json_schema(),
                },
            )
            summary_news = processes_schemas.SummaryNewsByLLM.model_validate_json(response.text)
            news_items_list_selected[i]["summary"] = summary_news.summary
        except Exception as e:
            print(f"  [Gemini] 오류: {e}")

    print(f"DEBUG: Gemini 뉴스 요약 완료.")
    return news_items_list_selected