import os
import json
import asyncio
import pymysql 
from celery import Celery
from dotenv import load_dotenv
from typing import List, Dict, Optional
from urllib.parse import urlparse 
from kombu import Queue 
from celery.result import allow_join_result # [신규] 이 줄을 꼭 추가하세요!

from services import processes_svc 

load_dotenv()

# --- 환경 변수 로드 ---
DB_CONN = os.getenv("DB_CONN")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

if not DB_CONN:
    raise ValueError("DB_CONN 환경 변수가 .env 파일에 설정되지 않았습니다.")

# --- Celery 앱 설정 ---
os.environ.setdefault('FORKED_BY_MULTIPROCESSING', '1') 

celery_app = Celery(
    'tasks',
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=['celery_worker'] 
)

# --- 큐(Queue) 및 라우팅 설정 ---
celery_app.conf.update(
    timezone='Asia/Seoul',
    enable_utc=True,
    task_queues = (
        Queue('cpu_io', default=True), 
        Queue('gpu'),                  
    ),
    task_default_queue = 'cpu_io', 
    task_routes = {
        'celery_worker.run_sbert_task': {'queue': 'gpu'}
    }
)

# --- 동기 DB 연결 ---
def _parse_db_conn(db_conn_str: str) -> dict:
    try:
        if db_conn_str.startswith("mysql+aiomysql://"):
            db_conn_str = "mysql://" + db_conn_str[len("mysql+aiomysql://"):]
        parsed = urlparse(db_conn_str)
        return {
            "host": parsed.hostname,
            "port": parsed.port or 3306,
            "user": parsed.username,
            "password": parsed.password,
            "database": parsed.path.lstrip('/'),
            "charset": "utf8mb4",
            "cursorclass": pymysql.cursors.DictCursor
        }
    except Exception as e:
        print(f"DB_CONN 파싱 오류: {e}")
        raise ValueError("DB_CONN 환경 변수가 잘못되었습니다.")

DB_CONFIG = _parse_db_conn(DB_CONN)

def get_sync_db_connection():
    try:
        return pymysql.connect(**DB_CONFIG)
    except pymysql.Error as e:
        print(f"동기 DB 연결 실패: {e}")
        return None

# --- [GPU Task] S-BERT 전용 ---
@celery_app.task(name='celery_worker.run_sbert_task') 
def run_sbert_task(
    summary_meeting: str, 
    news_items_list: List[Dict[str, Optional[str]]]
) -> List[Dict[str, Optional[str]]]:
    print("  [GPU Task] S-BERT 작업 수신. 분석 시작...")
    try:
        result_list = processes_svc._run_sbert_similarity(
            summary_meeting,
            news_items_list
        )
        print("  [GPU Task] S-BERT 분석 완료.")
        return result_list
    except Exception as e:
        print(f"  [GPU Task] S-BERT 작업 실패: {e}")
        return news_items_list[:processes_svc.TOP_N_NEWS]


# --- [CPU/IO Task] 메인 작업 ---
@celery_app.task(name='process_news_task', bind=True)
def process_news_task(
    self, 
    meeting_id: int, 
    user_id: int, 
    summary_meeting: str, 
    keyword_meeting_list: List[str]
):
    print(f"[CPU/IO Task] 작업 시작: meeting_id = {meeting_id}")

    try:
        # 1. Google News URL 검색
        print(f"  [Step 1] Google News URL 검색 (키워드: {keyword_meeting_list})")
        news_url_list = asyncio.run(
            processes_svc.news_url_list_by_custom_search_json_api(keyword_meeting_list)
        )
        if not news_url_list:
            return f"작업 완료 (meeting_id: {meeting_id}): 검색된 뉴스 없음."
        print(f"  [Step 1] URL {len(news_url_list)}개 검색 완료.")

        # 2. 뉴스 크롤링
        print(f"  [Step 2] Newspaper3k 크롤링 시작 (URL {len(news_url_list)}개)")
        news_items_list = asyncio.run(
            processes_svc.original_news_list_by_newspaper3k(news_url_list)
        )
        if not news_items_list:
            return f"작업 완료 (meeting_id: {meeting_id}): 크롤링된 뉴스 없음."
        print(f"  [Step 2] 크롤링 성공 (총 {len(news_items_list)}개).")

        # 3. [수정] S-BERT 분석 (허락받고 기다리기)
        print(f"  [Step 3] S-BERT 작업을 'gpu' 큐로 전송 및 대기...")
        try:
            # [수정] allow_join_result()를 사용하여 대기 허용
            with allow_join_result():
                sbert_result = run_sbert_task.delay(summary_meeting, news_items_list).get(timeout=300)
            
            news_items_list_selected = sbert_result
            print(f"  [Step 3] S-BERT 작업 완료. GPU 워커로부터 결과 수신.")
        
        except Exception as e:
            print(f"  [Step 3] S-BERT 서브태스크 호출 실패: {e}")
            print("  [Step 3] S-BERT Fallback: 크롤링 순서대로 5개 사용")
            news_items_list_selected = news_items_list[:processes_svc.TOP_N_NEWS]

        # 4. 선별된 뉴스 요약
        print(f"  [Step 4] Gemini 뉴스 요약 시작 (뉴스 {len(news_items_list_selected)}개)")
        news_items_list_final = asyncio.run(
            processes_svc.summary_news_list_by_gemini(news_items_list_selected)
        )
        print(f"  [Step 4] 뉴스 요약 완료.")

        # 5. 최종 DB 업데이트
        print(f"  [Step 5] DB 업데이트 시작 (meeting_id: {meeting_id})")
        
        conn = None
        cursor = None
        try:
            conn = get_sync_db_connection()
            if conn is None:
                raise Exception("동기 DB 연결 실패")
            cursor = conn.cursor()
            query = "UPDATE meetings SET news_items = %s WHERE id = %s"
            news_items_json = json.dumps(news_items_list_final, ensure_ascii=False)
            cursor.execute(query, (news_items_json, meeting_id))
            conn.commit()
            print(f"  [Step 5] DB 업데이트 성공.")
        except Exception as e:
            print(f"  [Step 5] DB 업데이트 오류: {e}")
            if conn: conn.rollback()
            raise 
        finally:
            if cursor: cursor.close()
            if conn: conn.close()

        print(f"[CPU/IO Task] 작업 완료: meeting_id = {meeting_id}")
        return f"작업 완료 (meeting_id: {meeting_id}): 뉴스 {len(news_items_list_final)}개 업데이트"

    except Exception as e:
        print(f"[CPU/IO Task] 작업 실패 (meeting_id: {meeting_id}): {e}")
        raise self.retry(exc=e, countdown=180)


if __name__ == '__main__':
    pass