from app.base.exception.exception import show_log
from app.src.v1.backend.api import send_progress_task
from app.src.v1.lora_trainner.lora_trainner import lora_trainner
from app.src.v1.lora_trainner.repo_init_resource_files import init_resource_files_from_urls
from app.src.v1.schemas.base import LoraTrainnerRequest, SendProgressTaskRequest, SDXLRequest, SDRequest, T2SRequest, MusicGenRequest, AudioGenRequest, GTERequest
from app.src.v1.sd.sd import sd
from app.src.v1.sdxl.sdxl import sdxl
from app.src.v1.sdxl.sdxl import sdxl_lightning
from app.src.v1.txt2vid.txt2vid import txt2vid
from app.src.v1.llm.text_completion import text_completion
from app.src.v1.bark.bark_txt2speech import text2speech
from app.src.v1.music_gen.music import music
from app.src.v1.audio_gen.audio import audio


def worker_lora_trainner(
        celery_task_id: str,
        celery_task_name: str,
        request_data: LoraTrainnerRequest,
):
    show_log(
        message="function: worker_lora_trainner, "
                f"celery_task_id: {celery_task_id}, "
                f"celery_task_name: {celery_task_name}"
    )

    input_paths, output_paths, minio_output_paths = init_resource_files_from_urls(
        prompt=request_data['prompt'],
        file_urls=request_data['minio_input_paths']
    )

    is_success, response, error = lora_trainner(
        celery_task_id=celery_task_id,
        request_data=request_data,
        input_paths=input_paths,
        output_paths=output_paths,
        minio_output_paths=minio_output_paths,
    )

    return {
        "is_success": is_success,
        "response": response,
        "error": error
    }


def worker_sdxl(
        celery_task_id: str,
        celery_task_name: str,
        request_data: SDXLRequest,
):
    show_log(
        message="function: worker_sdxl, "
                f"celery_task_id: {celery_task_id}, "
                f"celery_task_name: {celery_task_name}"
    )

    send_progress_task(
        SendProgressTaskRequest(
            task_id=request_data['task_id'],
            task_type="SDXL",
            percent=10
        )
    )
    is_success, response, error = sdxl(
        celery_task_id=celery_task_id,
        request_data=request_data,
    )
    return {
        "is_success": is_success,
        "response": response,
        "error": error
    }


def worker_sdxl_lightning(
        celery_task_id: str,
        celery_task_name: str,
        request_data: SDXLRequest,
):
    show_log(
        message="function: worker_sdxl_lightning, "
                f"celery_task_id: {celery_task_id}, "
                f"celery_task_name: {celery_task_name}"
    )

    send_progress_task(
        SendProgressTaskRequest(
            task_id=request_data['task_id'],
            task_type="SDXL",
            percent=10
        )
    )
    is_success, response, error = sdxl_lightning(
        celery_task_id=celery_task_id,
        request_data=request_data,
    )
    return {
        "is_success": is_success,
        "response": response,
        "error": error
    }


def worker_sd(
        celery_task_id: str,
        celery_task_name: str,
        request_data: SDRequest,
):
    show_log(
        message="function: worker_sd, "
                f"celery_task_id: {celery_task_id}, "
                f"celery_task_name: {celery_task_name}"
    )

    send_progress_task(
        SendProgressTaskRequest(
            task_id=request_data['task_id'],
            task_type="SD",
            percent=10
        )
    )
    is_success, response, error = sd(
        celery_task_id=celery_task_id,
        request_data=request_data,
    )
    
    return {
        "is_success": is_success,
        "response": response,
        "error": error
    }


def worker_txt2vid(
        celery_task_id: str,
        celery_task_name: str,
        request_data: SDRequest,
):
    show_log(
        message="function: worker_txt2vid"
                f"celery_task_id: {celery_task_id}, "
                f"celery_task_name: {celery_task_name}"
    )

    send_progress_task(
        SendProgressTaskRequest(
            task_id=request_data['task_id'],
            task_type="TXT2VID",
            percent=10
        )
    )
    is_success, response, error = txt2vid(
        celery_task_id=celery_task_id,
        request_data=request_data,
    )
    
    return {
        "is_success": is_success,
        "response": response,
        "error": error
    }


def worker_text_completion(
        celery_task_id: str,
        celery_task_name: str,
        request_data: SDRequest,
):
    show_log(
        message="function: worker_text_completion"
                f"celery_task_id: {celery_task_id}, "
                f"celery_task_name: {celery_task_name}"
    )

    is_success, response, error = text_completion(
        celery_task_id=celery_task_id,
        request_data=request_data,
    )
    
    return {
        "is_success": is_success,
        "response": response,
        "error": error
    }


def worker_t2s(
        celery_task_id: str,
        celery_task_name: str,
        request_data: T2SRequest,
):
    show_log(
        message="function: worker_t2s"
                f"celery_task_id: {celery_task_id}, "
                f"celery_task_name: {celery_task_name}"
    )

    send_progress_task(
        SendProgressTaskRequest(
            task_id=request_data['task_id'],
            task_type="T2S",
            percent=10
        )
    )
    is_success, response, error = text2speech(
        celery_task_id=celery_task_id,
        request_data=request_data,
    )
    
    return {
        "is_success": is_success,
        "response": response,
        "error": error
    }


def worker_musicgen(
        celery_task_id: str,
        celery_task_name: str,
        request_data: MusicGenRequest,
):
    show_log(
        message="function: worker_musicgen"
                f"celery_task_id: {celery_task_id}, "
                f"celery_task_name: {celery_task_name}"
    )
    
    is_success, response, error = music(
        celery_task_id=celery_task_id,
        request_data=request_data,
    )
    
    return {
        "is_success": is_success,
        "response": response,
        "error": error
    }

def worker_audiogen(
        celery_task_id: str,
        celery_task_name: str,
        request_data: AudioGenRequest,
):
    show_log(
        message="function: worker_audio_gen"
                f"celery_task_id: {celery_task_id}, "
                f"celery_task_name: {celery_task_name}"
    )

    is_success, response, error = audio(
        celery_task_id=celery_task_id,
        request_data=request_data,
    )
    
    return {
        "is_success": is_success,
        "response": response,
        "error": error
    }

def worker_gte(
        celery_task_id: str,
        celery_task_name: str,
        request_data: GTERequest,
):
    show_log(
        message="function: worker_gte"
                f"celery_task_id: {celery_task_id}, "
                f"celery_task_name: {celery_task_name}"
    )

    send_progress_task(
        SendProgressTaskRequest(
            task_id=request_data['task_id'],
            task_type="GTE",
            percent=10
        )
    )
    is_success, response, error = audio(
        celery_task_id=celery_task_id,
        request_data=request_data,
    )
    
    return {
        "is_success": is_success,
        "response": response,
        "error": error
    }