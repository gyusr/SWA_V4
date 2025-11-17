from sqlalchemy import Connection, text
from sqlalchemy.exc import SQLAlchemyError
from fastapi.exceptions import HTTPException
from fastapi import status
from typing import Dict
from services import processes_svc
import json
from schemas import meetings_schemas

# [추가] Celery 태스크 임포트
try:
    from celery_worker import process_news_task
except ImportError:
    # Celery가 설치되지 않았거나 경로 문제가 있을 경우
    print("경고: Celery 태스크를 임포트할 수 없습니다. 백그라운드 작업이 비활성화됩니다.")
    process_news_task = None
except Exception as e:
    print(f"경고: Celery 임포트 중 알 수 없는 오류 발생: {e}")
    process_news_task = None


# [수정] create_meeting 함수 (즉각 응답 + Celery 호출)
async def create_meeting(
    conn: Connection,
    title: str,
    original_meeting: str,
    session_user: Dict
):
    # 1. original_meeting -> summary_meeting_text / keyword_meeting_list (즉시 실행)
    print(f"DEBUG: (FastAPI) Step 1. Gemini 회의록 요약 시작...")
    try:
        summary_meeting, keyword_meeting_list = await processes_svc.summary_meeting_and_keyword_meeting_by_gemini(original_meeting=original_meeting)
        print(f"DEBUG: (FastAPI) Step 1. Gemini 회의록 요약 완료.")
    except Exception as e:
        print(f"Gemini 요약 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Gemini 회의록 요약 중 오류 발생: {e}"
        )

    # (제거) 2~5. 뉴스 관련 로직 (Celery로 이동)

    # 2. [수정] 1차 DB 저장 (news_items는 NULL, keywords는 저장)
    print(f"DEBUG: (FastAPI) Step 2. 1차 DB 저장 시작 (news_items=NULL)...")
    meeting_id = None
    try:
        query = '''
        insert into meetings(
        user_id, title, created_dt,
        original_meeting, summary_meeting, keywords
        )
        values(
        :user_id, :title, now(),
        :original_meeting, :summary_meeting, :keywords
        )
        '''

        result = await conn.execute(
            text(query),
            {
                "user_id": session_user["id"],
                "title": title,
                "original_meeting": original_meeting,
                "summary_meeting": summary_meeting,
                "keywords": json.dumps(keyword_meeting_list, ensure_ascii=False),
                # "news_items"는 제외 (DB 기본값 NULL 적용)
            }
        )
        
        # [추가] 방금 INSERT된 ID 가져오기
        meeting_id = result.lastrowid
        await conn.commit()
        print(f"DEBUG: (FastAPI) Step 2. 1차 DB 저장 완료 (New meeting_id: {meeting_id})")

    except SQLAlchemyError as e:
        print(e)
        await conn.rollback() # 롤백
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="서버 내부 오류 (DB 저장 실패)"
        )
    except Exception as e:
        print(e)
        await conn.rollback() # 롤백
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="알 수 없는 오류 (DB 저장 실패)"
        )

    # 3. [추가] Celery 백그라운드 태스크 호출
    if process_news_task and meeting_id:
        print(f"DEBUG: (FastAPI) Step 3. Celery 태스크 호출 (process_news_task.delay)...")
        try:
            process_news_task.delay(
                meeting_id=meeting_id,
                user_id=session_user["id"],
                summary_meeting=summary_meeting,
                keyword_meeting_list=keyword_meeting_list
            )
            print(f"DEBUG: (FastAPI) Step 3. Celery 태스크 호출 완료.")
        except Exception as e:
            # Celery 호출 실패 (예: Redis 연결 불가)
            # 이 경우 DB는 1차 저장되었지만, 백그라운드 작업은 실행되지 않음.
            # (중요 로깅 필요)
            print(f"경고: Celery 태스크 호출 실패! (meeting_id: {meeting_id}): {e}")
            # 하지만 사용자는 이미 즉시 응답(리디렉션)을 받아야 하므로 오류를 raise하지 않음.
    
    elif not process_news_task:
         print(f"경고: Celery 태스크(process_news_task)가 정의되지 않아 백그라운드 작업을 건너뜁니다.")

    # 4. (FastAPI) 즉시 리디렉션
    # (이 함수가 반환되면 routes/meetings.py에서 리디렉션 수행)


async def get_all_meetings(
    conn: Connection,
    session_user: Dict
):
    all_meetings = []

    try:
        query = '''
        select id, title, created_dt from meetings
        where user_id = :user_id
        order by created_dt desc
        '''
        # [수정] 컬럼명 명시, 정렬 추가

        result = await conn.execute(
            text(query),
            {
                "user_id": session_user["id"]
            }
        )

        rows = result.fetchall()
        if rows is None:
            return all_meetings
        
        all_meetings = [meetings_schemas.MeetingData(
            id=row.id,
            title=row.title,
            created_dt=row.created_dt
        ) for row in rows]

        return all_meetings

    except SQLAlchemyError as e:
        print(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    
async def get_by_id_meeting(
    conn: Connection,
    id: int,
    session_user: Dict
):
    try:
        query ='''
        select * from meetings
        where user_id = :user_id and id = :id
        '''

        result = await conn.execute(
            text(query),
            {
                "user_id": session_user["id"],
                "id": id
            }
        )

        row = result.fetchone()
        
        # [추가] row가 없을 경우 (잘못된 ID 또는 권한 없음)
        if row is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="해당 회의록을 찾을 수 없거나 접근 권한이 없습니다."
            )
            
        meeting_data = row._asdict()

        # --- 4. [수정] Pydantic Optional 스키마에 맞게 NULL(None) 처리 ---
        try:
            # news_items가 None이거나 빈 문자열이 아니면서, 문자열일 경우에만 json.loads
            if meeting_data.get('news_items') and isinstance(meeting_data['news_items'], str):
                meeting_data['news_items'] = json.loads(meeting_data['news_items'])
            # None이거나, 이미 파싱되었거나, 빈 문자열이면 -> None으로 설정 (스키마가 Optional[List]=None 이므로)
            elif not meeting_data.get('news_items'):
                meeting_data['news_items'] = None # Pydantic이 None을 처리

            # keywords도 동일하게 처리
            if meeting_data.get('keywords') and isinstance(meeting_data['keywords'], str):
                meeting_data['keywords'] = json.loads(meeting_data['keywords'])
            elif not meeting_data.get('keywords'):
                meeting_data['keywords'] = None # Pydantic이 None을 처리
        
        except json.JSONDecodeError as e:
            print(f"JSON Parse Error (meeting_id: {id}): {e}")
            # 파싱 실패 시에도 None으로 설정하여 Pydantic 오류 방지
            meeting_data['news_items'] = None
            meeting_data['keywords'] = None
            # (중요 로깅 필요)

        # --- 5. (FIX) Validate using the *modified* dict ---
        try:
            meeting = meetings_schemas.MeetingDataByID(**meeting_data)
        except Exception as e:
            print(f"Pydantic 유효성 검사 실패 (meeting_id: {id}): {e}")
            print(f"  -> 데이터: {meeting_data}")
            raise HTTPException(
                status_code=500, detail="데이터 유효성 검사 실패."
            )

        return meeting
            

    except SQLAlchemyError as e:
        print(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

# [신규] 재시도를 위해 news_items를 NULL로 비우는 함수
async def clear_news_items_by_id(
    conn: Connection,
    id: int,
    session_user: Dict
):
    """뉴스 재시도를 위해 news_items 컬럼을 NULL로 설정합니다."""
    try:
        query = """
        UPDATE meetings
        SET news_items = NULL
        WHERE id = :id AND user_id = :user_id
        """
        
        await conn.execute(
            text(query),
            {
                "id": id,
                "user_id": session_user["id"]
            }
        )
        await conn.commit()
        print(f"DEBUG: (FastAPI) news_items 컬럼 NULL로 초기화 (meeting_id: {id})")

    except SQLAlchemyError as e:
        print(e)
        await conn.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="DB 업데이트 중 오류 발생 (뉴스 초기화 실패)"
        )