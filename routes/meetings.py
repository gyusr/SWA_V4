from fastapi import APIRouter, Request, status, Depends, Form
from fastapi.responses import RedirectResponse, JSONResponse # [수정] JSONResponse 임포트
from fastapi.templating import Jinja2Templates
from fastapi.exceptions import HTTPException 

from sqlalchemy import Connection
from db.database import context_get_conn

from services import meetings_svc, user_svc, processes_svc

# [추가] Celery 태스크 임포트
try:
    from celery_worker import process_news_task
except ImportError:
    process_news_task = None
except Exception as e:
    print(f"경고: Celery 임포트 중 알 수 없는 오류 발생: {e}")
    process_news_task = None

router = APIRouter(prefix="/meetings", tags=["meetings"])
templates = Jinja2Templates(directory="templates")


# 회의록 생성
@router.get("/create")
async def create_meeting_ui(
    request: Request,
    session_user = Depends(user_svc.get_session_user_prt)
):
    return templates.TemplateResponse(
        request=request,
        name="create_meeting.html",
        context={
            "session_user": session_user
        }
    )

@router.post("/create")
async def create_meeting(
    request: Request,
    title: str = Form(...), # 조건
    original_meeting: str = Form(...), # 조건
    conn: Connection = Depends(context_get_conn),
    session_user = Depends(user_svc.get_session_user_prt)
):
    # 제일 중요한 로직!!!!!
    await meetings_svc.create_meeting(
        conn=conn,
        title=title,
        original_meeting=original_meeting,
        session_user=session_user
    )
    
    return RedirectResponse(
        url="/meetings/read/all",
        status_code=status.HTTP_303_SEE_OTHER
    )


# 회의록 읽기
@router.get("/read/all")
async def get_all_meetings_ui(
    request: Request,
    conn: Connection = Depends(context_get_conn),
    session_user = Depends(user_svc.get_session_user_prt)
):
    # 모든 회의록 가져오기 (id / title / created_dt) - 본인 것만 가져와야함 / 세션 기반으로
    all_meetings = await meetings_svc.get_all_meetings(
        conn=conn,
        session_user=session_user
    )

    return templates.TemplateResponse(
        request=request,
        name="main_meeting.html",
        context={
            "all_meetings": all_meetings,
            "session_user": session_user
        }
    )


@router.get("/read/{id}")
async def get_by_id_meeting_ui(
    request: Request,
    id: int,
    conn: Connection = Depends(context_get_conn),
    session_user = Depends(user_svc.get_session_user_prt)
):
    # id 기반 회의록 가져오기 / 세션 기반으로
    meeting = await meetings_svc.get_by_id_meeting(
        conn=conn,
        id=id,
        session_user=session_user
    )
    
    return templates.TemplateResponse(
        request=request,
        name="read_meeting.html",
        context={
            "meeting": meeting,
            "session_user": session_user
        }
    )


# 회의록 수정 - 필요 없을 듯??


# 회의록 삭제
@router.delete("/delete/{id}")
async def delete_meeting(
    request: Request,
    id: int,
    conn: Connection = Depends(context_get_conn),
    session_user = Depends(user_svc.get_session_user_prt)
): 
    # 회의록 삭제 로직
    # (TODO: 삭제 로직을 services/meetings_svc.py에 구현해야 함)
    # await meetings_svc.delete_meeting_by_id(conn=conn, id=id, session_user=session_user)

    return RedirectResponse(
        url="/meetings/read/all", # [수정] /meetings -> /meetings/read/all
        status_code=status.HTTP_303_SEE_OTHER
    )


# [신규] 뉴스 분석 재시도
@router.post("/retry-news/{id}")
async def retry_news_analysis(
    request: Request,
    id: int,
    conn: Connection = Depends(context_get_conn),
    session_user = Depends(user_svc.get_session_user_prt)
):
    if not process_news_task:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Celery 워커가 연결되지 않아 재시도할 수 없습니다."
        )

    # 1. 재시도할 미팅의 키워드/요약본을 DB에서 가져옴
    meeting = await meetings_svc.get_by_id_meeting(
        conn=conn,
        id=id,
        session_user=session_user
    )

    if not meeting.keywords or not meeting.summary_meeting:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="키워드 또는 요약본이 없어 재시도할 수 없습니다."
        )
        
    # 2. [신규] news_items 컬럼을 NULL로 비워서 "로딩" 상태로 만듦
    await meetings_svc.clear_news_items_by_id(
        conn=conn,
        id=id,
        session_user=session_user
    )

    # 3. [수정] (2->3) Celery 태스크(process_news_task)를 다시 호출
    print(f"DEBUG: (FastAPI) 뉴스 분석 재시도 시작 (meeting_id: {id})")
    try:
        process_news_task.delay(
            meeting_id=meeting.id,
            user_id=session_user["id"],
            summary_meeting=meeting.summary_meeting,
            keyword_meeting_list=meeting.keywords
        )
    except Exception as e:
        print(f"경고: Celery 태스크 재시도 호출 실패! (meeting_id: {id}): {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Celery 작업 호출에 실패했습니다: {e}"
        )

    # 4. [수정] RedirectResponse 대신 JSONResponse 반환 (202 = Accepted)
    return JSONResponse(
        content={"status": "retry_started", "meeting_id": id},
        status_code=status.HTTP_202_ACCEPTED
    )