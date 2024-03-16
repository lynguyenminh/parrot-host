import os
from pathlib import Path

from celery import Celery
from celery.states import FAILURE, SUCCESS

from app.src.v1.backend.api import send_done_task, send_fail_task
from app.src.v1.schemas.base import (SendDoneTaskRequest, SendFailTaskRequest)
from app.src.v1.workers import worker_lora_trainner, worker_sd, worker_sdxl, worker_sdxl_lightning, worker_txt2vid, worker_text_completion, worker_t2s, worker_musicgen, worker_audiogen, worker_gte

from app.base.exception.exception import show_log

celery_app = Celery(
    os.environ['CELERY_APP_NAME'],
    broker=f"amqp://"
           f"{os.environ['CELERY_SERVICE_WORKER_USERNAME']}:{os.environ['CELERY_SERVICE_WORKER_PASSWORD']}"
           f"@{os.environ['CELERY_SERVICE_WORKER_HOST']}:{os.environ['CELERY_SERVICE_WORKER_PORT']}//",
    broker_connection_retry_on_startup=True,
    result_expires=int(os.environ['CELERY_RESULT_EXPIRES'])
)

Path("resources/input/images").mkdir(parents=True, exist_ok=True)
Path("resources/output/images").mkdir(parents=True, exist_ok=True)


def parrot_lora_trainer_task(self, request_data):
    result = None
    try:
        result = worker_lora_trainner(
            request_data=request_data,
            celery_task_id=self.request.id,
            celery_task_name=self.name,
        )
        if not result.get('is_success'):
            raise Exception("result is None")
        self.update_state(state=SUCCESS, meta={'result': result})
        is_success, _, _ = send_done_task(
            SendDoneTaskRequest(
                task_id=request_data['task_id'],
                task_type="LORA_TRAINNER",
            )
        )
        if not is_success:
            raise Exception("send done task failed")
    except Exception as ex:
        if self.request.retries >= self.max_retries:
            show_log(message=ex, level="error")
            self.update_state(state=FAILURE, meta={'result': result})
            is_success, _, _ = send_fail_task(
                SendFailTaskRequest(
                    task_id=request_data['task_id'],
                    task_type="LORA_TRAINNER",
                )
            )
            if not is_success:
                raise Exception("send fail task failed")
        else:
            show_log(
                message=f"Retry celery_id: {self.request.id},"
                        f" celery_task_name: {self.name},"
                        f" index: {self.request.retries}"
            )
            self.retry(exc=ex, countdown=int(os.environ['CELERY_RETRY_DELAY_TIME']))

def parrot_sdxl_lightning_task(self, request_data):
    result = None
    try:
        result = worker_sdxl_lightning(
            request_data=request_data,
            celery_task_id=self.request.id,
            celery_task_name=self.name,
        )
        if not result.get('is_success'):
            raise Exception("result is None")
        self.update_state(state=SUCCESS, meta={'result': result})
        is_success, _, _ = send_done_task(
            SendDoneTaskRequest(
                task_id=request_data['task_id'],
                task_type="SDXL_LIGHTNING",
            )
        )
        if not is_success:
            raise Exception("send done task failed")
    except Exception as ex:
        if self.request.retries >= self.max_retries:
            show_log(message=ex, level="error")
            self.update_state(state=FAILURE, meta={'result': result})
            is_success, _, _ = send_fail_task(
                SendFailTaskRequest(
                    task_id=request_data['task_id'],
                    task_type="SDXL_LIGHTNING",
                )
            )
            if not is_success:
                raise Exception("send fail task failed")
        else:
            show_log(
                message=f"Retry celery_id: {self.request.id},"
                        f" celery_task_name: {self.name},"
                        f" index: {self.request.retries}"
            )
            self.retry(exc=ex, countdown=int(os.environ['CELERY_RETRY_DELAY_TIME']))

def parrot_sdxl_task(self, request_data):
    result = None
    try:
        result = worker_sdxl(
            request_data=request_data,
            celery_task_id=self.request.id,
            celery_task_name=self.name,
        )
        if not result.get('is_success'):
            raise Exception("result is None")
        self.update_state(state=SUCCESS, meta={'result': result})
        is_success, _, _ = send_done_task(
            SendDoneTaskRequest(
                task_id=request_data['task_id'],
                task_type="SDXL",
            )
        )
        if not is_success:
            raise Exception("send done task failed")
    except Exception as ex:
        if self.request.retries >= self.max_retries:
            show_log(message=ex, level="error")
            self.update_state(state=FAILURE, meta={'result': result})
            is_success, _, _ = send_fail_task(
                SendFailTaskRequest(
                    task_id=request_data['task_id'],
                    task_type="SDXL",
                )
            )
            if not is_success:
                raise Exception("send fail task failed")
        else:
            show_log(
                message=f"Retry celery_id: {self.request.id},"
                        f" celery_task_name: {self.name},"
                        f" index: {self.request.retries}"
            )
            self.retry(exc=ex, countdown=int(os.environ['CELERY_RETRY_DELAY_TIME']))

def parrot_sd_task(self, request_data):
    result = None
    try:
        result = worker_sd(
            request_data=request_data,
            celery_task_id=self.request.id,
            celery_task_name=self.name,
        )
        if not result.get('is_success'):
            raise Exception("result is None")
        self.update_state(state=SUCCESS, meta={'result': result})
        is_success, _, _ = send_done_task(
            SendDoneTaskRequest(
                task_id=request_data['task_id'],
                task_type="SD",
            )
        )
        if not is_success:
            raise Exception("send done task failed")
    except Exception as ex:
        if self.request.retries >= self.max_retries:
            show_log(message=ex, level="error")
            self.update_state(state=FAILURE, meta={'result': result})
            is_success, _, _ = send_fail_task(
                SendFailTaskRequest(
                    task_id=request_data['task_id'],
                    task_type="SD",
                )
            )
            if not is_success:
                raise Exception("send fail task failed")
        else:
            show_log(
                message=f"Retry celery_id: {self.request.id},"
                        f" celery_task_name: {self.name},"
                        f" index: {self.request.retries}"
            )
            self.retry(exc=ex, countdown=int(os.environ['CELERY_RETRY_DELAY_TIME']))

def parrot_txt2vid_damo_task(self, request_data):
    result = None
    try:
        result = worker_txt2vid(
            request_data=request_data,
            celery_task_id=self.request.id,
            celery_task_name=self.name,
        )
        if not result.get('is_success'):
            raise Exception("result is None")
        self.update_state(state=SUCCESS, meta={'result': result})
    except Exception as ex:
        if self.request.retries >= self.max_retries:
            show_log(message=ex, level="error")
            self.update_state(state=FAILURE, meta={'result': result})
            is_success, _, _ = send_fail_task(
                SendFailTaskRequest(
                    task_id=request_data['task_id'],
                    task_type="TXT2VID",
                )
            )
            if not is_success:
                raise Exception("send fail task failed")
        else:
            show_log(
                message=f"Retry celery_id: {self.request.id},"
                        f" celery_task_name: {self.name},"
                        f" index: {self.request.retries}"
            )
            self.retry(exc=ex, countdown=int(os.environ['CELERY_RETRY_DELAY_TIME']))

def parrot_llm_gemma_7b_task(self, request_data):
    result = None
    try:
        result = worker_text_completion(
            request_data=request_data,
            celery_task_id=self.request.id,
            celery_task_name=self.name,
        )
        if not result.get('is_success'):
            raise Exception("result is None")
        self.update_state(state=SUCCESS, meta={'result': result})
    except Exception as ex:
        if self.request.retries >= self.max_retries:
            show_log(message=ex, level="error")
            self.update_state(state=FAILURE, meta={'result': result})
        else:
            show_log(
                message=f"Retry celery_id: {self.request.id},"
                        f" celery_task_name: {self.name},"
                        f" index: {self.request.retries}"
            )
            self.retry(exc=ex, countdown=int(os.environ['CELERY_RETRY_DELAY_TIME']))



def parrot_t2s_task(self, request_data):
    result = None
    try:
        result = worker_t2s(
            request_data=request_data,
            celery_task_id=self.request.id,
            celery_task_name=self.name,
        )
        if not result.get('is_success'):
            raise Exception("result is None")
        self.update_state(state=SUCCESS, meta={'result': result})
        is_success, _, _ = send_done_task(
            SendDoneTaskRequest(
                task_id=request_data['task_id'],
                task_type="T2S",
            )
        )
        if not is_success:
            raise Exception("send done task failed")
    except Exception as ex:
        if self.request.retries >= self.max_retries:
            show_log(message=ex, level="error")
            self.update_state(state=FAILURE, meta={'result': result})
            is_success, _, _ = send_fail_task(
                SendFailTaskRequest(
                    task_id=request_data['task_id'],
                    task_type="T2S",
                )
            )
            if not is_success:
                raise Exception("send fail task failed")
        else:
            show_log(
                message=f"Retry celery_id: {self.request.id},"
                        f" celery_task_name: {self.name},"
                        f" index: {self.request.retries}"
            )
            self.retry(exc=ex, countdown=int(os.environ['CELERY_RETRY_DELAY_TIME']))


def parrot_musicgen_task(self, request_data):
    result = None
    try:
        result = worker_musicgen(
            request_data=request_data,
            celery_task_id=self.request.id,
            celery_task_name=self.name,
        )
        if not result.get('is_success'):
            raise Exception("result is None")
        self.update_state(state=SUCCESS, meta={'result': result})
        is_success, _, _ = send_done_task(
            SendDoneTaskRequest(
                task_id=request_data['task_id'],
                task_type="musicgen",
            )
        )
        if not is_success:
            raise Exception("send done task failed")
    except Exception as ex:
        if self.request.retries >= self.max_retries:
            show_log(message=ex, level="error")
            self.update_state(state=FAILURE, meta={'result': result})
            is_success, _, _ = send_fail_task(
                SendFailTaskRequest(
                    task_id=request_data['task_id'],
                    task_type="musicgen",
                )
            )
            if not is_success:
                raise Exception("send fail task failed")
        else:
            show_log(
                message=f"Retry celery_id: {self.request.id},"
                        f" celery_task_name: {self.name},"
                        f" index: {self.request.retries}"
            )
            self.retry(exc=ex, countdown=int(os.environ['CELERY_RETRY_DELAY_TIME']))


def parrot_audiogen_task(self, request_data):
    result = None
    try:
        result = worker_audiogen(
            request_data=request_data,
            celery_task_id=self.request.id,
            celery_task_name=self.name,
        )
        if not result.get('is_success'):
            raise Exception("result is None")
        self.update_state(state=SUCCESS, meta={'result': result})
        is_success, _, _ = send_done_task(
            SendDoneTaskRequest(
                task_id=request_data['task_id'],
                task_type="audiogen",
            )
        )
        if not is_success:
            raise Exception("send done task failed")
    except Exception as ex:
        if self.request.retries >= self.max_retries:
            show_log(message=ex, level="error")
            self.update_state(state=FAILURE, meta={'result': result})
            is_success, _, _ = send_fail_task(
                SendFailTaskRequest(
                    task_id=request_data['task_id'],
                    task_type="audiogen",
                )
            )
            if not is_success:
                raise Exception("send fail task failed")
        else:
            show_log(
                message=f"Retry celery_id: {self.request.id},"
                        f" celery_task_name: {self.name},"
                        f" index: {self.request.retries}"
            )
            self.retry(exc=ex, countdown=int(os.environ['CELERY_RETRY_DELAY_TIME']))


def parrot_gte_task(self, request_data):
    result = None
    try:
        result = worker_gte(
            request_data=request_data,
            celery_task_id=self.request.id,
            celery_task_name=self.name,
        )
        if not result.get('is_success'):
            raise Exception("result is None")
        self.update_state(state=SUCCESS, meta={'result': result})
        is_success, _, _ = send_done_task(
            SendDoneTaskRequest(
                task_id=request_data['task_id'],
                task_type="GTE",
            )
        )
        if not is_success:
            raise Exception("send done task failed")
    except Exception as ex:
        if self.request.retries >= self.max_retries:
            show_log(message=ex, level="error")
            self.update_state(state=FAILURE, meta={'result': result})
            is_success, _, _ = send_fail_task(
                SendFailTaskRequest(
                    task_id=request_data['task_id'],
                    task_type="GTE",
                )
            )
            if not is_success:
                raise Exception("send fail task failed")
        else:
            show_log(
                message=f"Retry celery_id: {self.request.id},"
                        f" celery_task_name: {self.name},"
                        f" index: {self.request.retries}"
            )
            self.retry(exc=ex, countdown=int(os.environ['CELERY_RETRY_DELAY_TIME']))