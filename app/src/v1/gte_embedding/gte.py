import time

from app.base.exception.exception import show_log
from app.src.v1.schemas.base import DoneGTERequest, GTERequest
from app.src.v1.backend.api import update_status_for_task, send_done_llm_task
from app.services.ai_services.text_completion import run_gte_large
from app.src.v1.schemas.base import UpdateStatusTaskRequest


def text_embedding(
        celery_task_id: str,
        request_data: GTERequest,
):
    show_log(
        message="function: text_embedding "
                f"celery_task_id: {celery_task_id}"
    )
    try:
        t0 = time.time()
        result = run_gte_large(request_data['text'], request_data['config']).tolist()
        t1 = time.time()
        show_log(f"Time processed: {t1-t0}")
        # update task status
        is_success, response, error = update_status_for_task(
            UpdateStatusTaskRequest(
                task_id=request_data['task_id'],
                status="COMPLETED",
                result=result
            )
        )
        if not response:
            show_log(
                message="function: text2speech, "
                f"celery_task_id: {celery_task_id}, "
                f"error: {error}"
            )
            return response
        
        send_done_llm_task(
            DoneGTERequest(
                task_id=request_data['task_id'],
                response=result
            )
        )
        return True, response, None
    
    except Exception as e:
        print(str(e))
        return False, None, str(e)