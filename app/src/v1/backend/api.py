import os

import requests

from app.base.exception.exception import show_log
from app.src.v1.schemas.base import (
    DoneLoraTrainnerRequest,
    DoneSDRequest, DoneSDXLRequest, DoneLLMRequest, DoneT2SRequest, DoneGTERequest,
    SendDoneTaskRequest,
    SendFailTaskRequest,
    SendProgressTaskRequest,
    UpdateStatusTaskRequest,
    DoneAudioGenRequest,
    DoneMusicGenRequest
)

from app.utils.services import JWT_TOKEN


def send_done_lora_trainner_task(request_data: DoneLoraTrainnerRequest):
    try:
        response = requests.post(
            headers={
                "Authorization": f"Bearer {JWT_TOKEN}",
                "Content-Type": "application/json"
            },
            url=f"{os.getenv('HOST_BACKEND_SERVICE')}/done_lora_trainner_task",
            json=request_data.dict()
        )
        if response.status_code == 200:
            return True, response.json(), None
        return False, None, response.json()
    except Exception as e:
        show_log(message=e, level="error")
        return False, None, e


def update_status_for_task(request_data: UpdateStatusTaskRequest):
    try:
        response = requests.post(
            headers={
                "Authorization": f"Bearer {JWT_TOKEN}",
                "Content-Type": "application/json"
            },
            url=f"{os.getenv('HOST_BACKEND_SERVICE')}/update_status_for_task",
            json=request_data.dict()
        )
        if response.status_code == 200:
            return True, response.json(), None
        return False, None, response.json()
    except Exception as e:
        show_log(message=e, level="error")
        return False, None, e


def send_fail_task(request_data: SendFailTaskRequest):
    try:
        response = requests.post(
            headers={
                "Authorization": f"Bearer {JWT_TOKEN}",
                "Content-Type": "application/json"
            },
            url=f"{os.getenv('HOST_BACKEND_SERVICE')}/fail_task",
            json=request_data.dict()
        )
        if response.status_code == 200:
            return True, response.json(), None
        return False, None, response.json()
    except Exception as e:
        show_log(message=e, level="error")
        return False, None, e


def send_done_task(request_data: SendDoneTaskRequest):
    try:
        response = requests.post(
            headers={
                "Authorization": f"Bearer {JWT_TOKEN}",
                "Content-Type": "application/json"
            },
            url=f"{os.getenv('HOST_BACKEND_SERVICE')}/done_task",
            json=request_data.dict()
        )
        if response.status_code == 200:
            return True, response.json(), None
        return False, None, response.json()
    except Exception as e:
        show_log(message=e, level="error")
        return False, None, e


def send_progress_task(request_data: SendProgressTaskRequest):
    try:
        response = requests.post(
            headers={
                "Authorization": f"Bearer {JWT_TOKEN}",
                "Content-Type": "application/json"
            },
            url=f"{os.getenv('HOST_BACKEND_SERVICE')}/progress_task",
            json=request_data.dict()
        )
        if response.status_code == 200:
            return True, response.json(), None
        return False, None, response.json()
    except Exception as e:
        show_log(message=e, level="error")
        return False, None, e


def send_done_sdxl_task(request_data: DoneSDXLRequest):
    try:
        response = requests.post(
            headers={
                "Authorization": f"Bearer {JWT_TOKEN}",
                "Content-Type": "application/json"
            },
            url=f"{os.getenv('HOST_BACKEND_SERVICE')}/done_sdxl_task",
            json=request_data.dict()
        )
        if response.status_code == 200:
            return True, response.json(), None
        return False, None, response.json()
    except Exception as e:
        show_log(message=e, level="error")
        return False, None, e


def send_done_sdxl_lightning_task(request_data: DoneSDXLRequest):
    try:
        response = requests.post(
            headers={
                "Authorization": f"Bearer {JWT_TOKEN}",
                "Content-Type": "application/json"
            },
            url=f"{os.getenv('HOST_BACKEND_SERVICE')}/done_sdxl_lightning_task",
            json=request_data.dict()
        )
        print(response.status_code, response.json())
        if response.status_code == 200:
            return True, response.json(), None
        return False, None, response.json()
    except Exception as e:
        show_log(message=e, level="error")
        return False, None, e


def send_done_sd_task(request_data: DoneSDRequest):
    try:
        response = requests.post(
            headers={
                "Authorization": f"Bearer {JWT_TOKEN}",
                "Content-Type": "application/json"
            },
            url=f"{os.getenv('HOST_BACKEND_SERVICE')}/done_sd_task",
            json=request_data.dict()
        )
        if response.status_code == 200:
            return True, response.json(), None
        return False, None, response.json()
    except Exception as e:
        show_log(message=e, level="error")
        return False, None, e


def send_done_txt2vid_task(request_data: DoneSDRequest):
    try:
        response = requests.post(
            headers={
                "Authorization": f"Bearer {JWT_TOKEN}",
                "Content-Type": "application/json"
            },
            url=f"{os.getenv('HOST_BACKEND_SERVICE')}/send_done_task",
            json=request_data.dict()
        )
        if response.status_code == 200:
            return True, response.json(), None
        return False, None, response.json()
    except Exception as e:
        show_log(message=e, level="error")
        return False, None, e


def send_done_llm_task(request_data: DoneLLMRequest):
    try:
        response = requests.post(
            headers={
                "Authorization": f"Bearer {JWT_TOKEN}",
                "Content-Type": "application/json"
            },
            url=f"{os.getenv('HOST_BACKEND_SERVICE')}/done_llm_task",
            json=request_data.dict()
        )
        if response.status_code == 200:
            return True, response.json(), None
        return False, None, response.json()
    except Exception as e:
        show_log(message=e, level="error")
        return False, None, e


def send_done_gte_task(request_data: DoneGTERequest):
    try:
        response = requests.post(
            headers={
                "Authorization": f"Bearer {JWT_TOKEN}",
                "Content-Type": "application/json"
            },
            url=f"{os.getenv('HOST_BACKEND_SERVICE')}/done_gte_task",
            json=request_data.dict()
        )
        if response.status_code == 200:
            return True, response.json(), None
        return False, None, response.json()
    except Exception as e:
        show_log(message=e, level="error")
        return False, None, e
    

def send_done_t2s_task(request_data: DoneT2SRequest):
    try:
        response = requests.post(
            headers={
                "Authorization": f"Bearer {JWT_TOKEN}",
                "Content-Type": "application/json"
            },
            url=f"{os.getenv('HOST_BACKEND_SERVICE')}/send_done_task",
            json=request_data.dict()
        )
        if response.status_code == 200:
            return True, response.json(), None

        return False, None, response.json()
    except Exception as e:
        show_log(message=e, level="error")
        return False, None, e

def send_done_musicgen_task(request_data: DoneMusicGenRequest):
    try:
        response = requests.post(
            headers={
                "Authorization": f"Bearer {JWT_TOKEN}",
                "Content-Type": "application/json"
            },
            url=f"{os.getenv('HOST_BACKEND_SERVICE')}/send_done_task",
            json=request_data.dict()
        )
        if response.status_code == 200:
            return True, response.json(), None

        return False, None, response.json()
    except Exception as e:
        show_log(message=e, level="error")
        return False, None, e

def send_done_audiogen_task(request_data: DoneAudioGenRequest):
    try:
        response = requests.post(
            headers={
                "Authorization": f"Bearer {JWT_TOKEN}",
                "Content-Type": "application/json"
            },
            url=f"{os.getenv('HOST_BACKEND_SERVICE')}/send_done_task",
            json=request_data.dict()
        )
        if response.status_code == 200:
            return True, response.json(), None
        
        return False, None, response.json()
    except Exception as e:
        show_log(message=e, level="error")
        return False, None, e

