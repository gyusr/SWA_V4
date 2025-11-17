import os
import json
import asyncio
import pymysql # 동기 DB 접근용
from celery import Celery
from dotenv import load_dotenv
from typing import List, Dict, Optional
from urllib.parse import urlparse # DB_CONN 파싱용

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" # (있으면 좋음)
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

# FastAPI 서비스 모듈 임포트 (비동기 함수 호출용)
# (주의: Celery 환경에서 FastAPI의 비동기 함수를 호출하려면 asyncio.run() 사용 필요)
from services import processes_svc 

# .env 파일 로드 (DB_CONN, API 키 등)
# celery_worker.py가 프로젝트 루트에 있다고 가정
load_dotenv()

# --- 환경 변수 로드 ---
DB_CONN = os.getenv("DB_CONN")
# .env 파일에 REDIS_URL="redis://localhost:6379/0" 항목이 필요합니다.
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0") # .env에 없으면 기본값

if not DB_CONN:
    raise ValueError("DB_CONN 환경 변수가 .env 파일에 설정되지 않았습니다.")

# --- Celery 앱 설정 ---
# Gevent/asyncio 호환성을 위해 billiard 비활성화 
os.environ.setdefault('FORKED_BY_MULTIPROCESSING', '1') 

celery_app = Celery(
    'tasks',
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=['celery_worker'] # 이 파일 자신을 태스크 모듈로 포함
)

# Celery 설정 (타임존 등)
celery_app.conf.update(
    timezone='Asia/Seoul',
    enable_utc=True,
)

# --- 동기 DB 연결 (Celery Task용) ---

def _parse_db_conn(db_conn_str: str) -> dict:
    """aiomysql DSN('mysql+aiomysql://...')을 pymysql이 사용할 수 있는 dict로 파싱"""
    try:
        # "mysql+aiomysql://" 부분을 제거
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
            "cursorclass": pymysql.cursors.DictCursor # 결과를 딕셔너리로 받음
        }
    except Exception as e:
        print(f"DB_CONN 파싱 오류: {e} (입력: {db_conn_str})")
        raise ValueError("DB_CONN 환경 변수가 잘못되었습니다.")

DB_CONFIG = _parse_db_conn(DB_CONN)

def get_sync_db_connection():
    """Celery 태스크용 동기 DB 커넥션 생성"""
    try:
        return pymysql.connect(**DB_CONFIG)
    except pymysql.Error as e:
        print(f"동기 DB 연결 실패: {e}")
        return None

# --- Celery 백그라운드 태스크 정의 ---

@celery_app.task(name='process_news_task', bind=True) # bind=True로 self 접근 (재시도용)
def process_news_task(
    self, # bind=True로 인해 self 인자 추가
    meeting_id: int, 
    user_id: int, # [롤백] user_id 인자 다시 추가
    summary_meeting: str, 
    keyword_meeting_list: List[str] # [롤백] news_url_list 대신 keyword_meeting_list를 받음
):
    """
    [Celery Task] 회의록 생성 후 무거운 뉴스 분석 작업을 백그라운드에서 처리합니다.
    (asyncio.run()을 사용하여 비동기 서비스 함수들을 호출)
    """
    print(f"[Celery Task] 작업 시작: meeting_id = {meeting_id}")

    try:
        # 1. [롤백] Google News URL 검색 (총 50개) - 비동기 실행
        print(f"  [Step 1] Google News URL 검색 (키워드: {keyword_meeting_list})")
        news_url_list = asyncio.run(
            processes_svc.news_url_list_by_custom_search_json_api(keyword_meeting_list)
        )
        if not news_url_list:
            print("  [Step 1] 검색된 뉴스 URL 없음. 작업 종료.")
            return f"작업 완료 (meeting_id: {meeting_id}): 검색된 뉴스 없음."
        print(f"  [Step 1] URL {len(news_url_list)}개 검색 완료.")

        # 2. [수정] (Step 2) 뉴스 크롤링 (병렬) - 비동기 실행
        print(f"  [Step 2] Newspaper3k 크롤링 시작 (URL {len(news_url_list)}개)")
        news_items_list = asyncio.run(
            processes_svc.original_news_list_by_newspaper3k(news_url_list)
        )
        if not news_items_list:
            print("  [Step 2] 크롤링된 뉴스 없음. 작업 종료.")
            return f"작업 완료 (meeting_id: {meeting_id}): 크롤링된 뉴스 없음."
        print(f"  [Step 2] 크롤링 성공 (총 {len(news_items_list)}개).")

        # 3. [수정] (Step 3) S-BERT 관련도 분석 (Top 5 선별) - 비동기 실행 (내부 to_thread)
        print(f"  [Step 3] S-BERT 관련도 분석 시작...")
        news_items_list_selected = asyncio.run(
            processes_svc.cosine_similarity_by_sbert(
                summary_meeting, 
                news_items_list
            )
        )
        print(f"  [Step 3] Top {len(news_items_list_selected)}개 뉴스 선별 완료.")

        # 4. [수정] (Step 4) 선별된 뉴스 요약 (LLM) - 비동기 실행
        print(f"  [Step 4] Gemini 뉴스 요약 시작 (뉴스 {len(news_items_list_selected)}개)")
        news_items_list_final = asyncio.run(
            processes_svc.summary_news_list_by_gemini(news_items_list_selected)
        )
        print(f"  [Step 4] 뉴스 요약 완료.")

        # 5. [수정] (Step 5) 최종 DB 업데이트 (동기 pymysql 사용)
        print(f"  [Step 5] DB 업데이트 시작 (meeting_id: {meeting_id})")
        
        conn = None
        cursor = None
        try:
            conn = get_sync_db_connection()
            if conn is None:
                raise Exception("동기 DB 연결 실패")
            
            cursor = conn.cursor()
            
            query = """
            UPDATE meetings
            SET news_items = %s
            WHERE id = %s
            """
            
            # news_items를 JSON 문자열로 변환 (ensure_ascii=False)
            news_items_json = json.dumps(news_items_list_final, ensure_ascii=False)
            
            cursor.execute(query, (news_items_json, meeting_id))
            conn.commit()
            
            print(f"  [Step 5] DB 업데이트 성공.")
            
        except Exception as e:
            print(f"  [Step 5] DB 업데이트 오류: {e}")
            if conn:
                conn.rollback() # 오류 발생 시 롤백
            raise # 오류를 다시 발생시켜 Celery가 재시도하도록 함
        
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

        print(f"[Celery Task] 작업 완료: meeting_id = {meeting_id}")
        return f"작업 완료 (meeting_id: {meeting_id}): 뉴스 {len(news_items_list_final)}개 업데이트"

    except Exception as e:
        print(f"[Celery Task] 작업 실패 (meeting_id: {meeting_id}): {e}")
        # Celery가 태스크를 재시도하도록 예외를 다시 발생시킴 (3분 후)
        raise self.retry(exc=e, countdown=180)


if __name__ == '__main__':
    # 이 파일은 직접 실행되지 않고, Celery 워커에 의해 임포트됩니다.
    # 실행 명령어: celery -A celery_worker.celery_app worker --loglevel=info
    pass