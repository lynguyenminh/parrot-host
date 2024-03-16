import io
import os
import time
import uuid
import requests
import mimetypes

from app.base.exception.exception import show_log
from app.services.ai_services.image_generation import run_lora_trainer
from app.src.v1.backend.api import (update_status_for_task, send_done_lora_trainner_task)
from app.src.v1.schemas.base import (DoneLoraTrainnerRequest, UpdateStatusTaskRequest, LoraTrainnerRequest)
from app.utils.services import minio_client

def create_uuid_string(): 
    random_uuid = uuid.uuid4()
    uuid_string = str(random_uuid)
    return uuid_string


def download_image(url: str, folder_path: str):
    response = requests.get(url)
    if response.status_code == 200:
        content_type = response.headers['Content-Type']
        extension = mimetypes.guess_extension(content_type)
        
        if extension is None:
            extension = '.jpg' 
        
        if os.path.exists(folder_path) is False:
            os.makedirs(folder_path)

        filename = f"{create_uuid_string()}{extension}"
        
        with open(os.path.join(folder_path, filename), 'wb') as file:
            file.write(response.content)
        print(f"Successed: {filename}")
        return filename
    else:
        print(f"Failed: {response.status_code}")
        return None


def create_dataset(request_data: LoraTrainnerRequest, folder_dataset: str): 
    if os.path.exists(folder_dataset) is False:
        os.makedirs(folder_dataset)

    for image, description in zip(request_data['minio_input_paths'], request_data['prompt']):
        # download image
        img_name = download_image(image, folder_dataset)

        # create label
        support_img = [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG", ".webp", ".bmp"]
        for i in support_img:
            if img_name.endswith(i):
                img_name = img_name.replace(i, ".txt")
                break
        with open(f"{folder_dataset}/{img_name}", "w") as file:
            file.write(description)


def lora_trainer(
        celery_task_id: str,
        request_data: LoraTrainnerRequest,
):
    show_log(
        message="function: lora_trainer, "
                f"celery_task_id: {celery_task_id}"
    )
    try:
        result = ''
        # 1. create dataset
        user_uuid = create_uuid_string()
        folder_dataset = f"app/services/ai_services/lora_trainer/tmp/{user_uuid}"
        create_dataset(request_data, folder_dataset)
        print(f"[INFO] Create dataset successfully: {folder_dataset}")


        # 2. gọi hàm train để train và lấy modelpath
        t0 = time.time()
        trainer_config = {
            "data_dir": os.path.join(os.getcwd(), folder_dataset),
            "user_name": user_uuid, 
            "sdxl": request_data.get("is_sdxl", "0")
        }

        model_path = run_lora_trainer(trainer_config)
        t1 = time.time()
        show_log(f"Time generated: {t1-t0}")
        print(f"[INFO] Train model successfully: {model_path}")

        # 3. Save the model to MinIO
        byte_buffer = io.BytesIO()
        with open(model_path, "rb") as file:
            byte_buffer.write(file.read())

        # Upload to MinIO
        s3_key = f"generated_result/{request_data['task_id']}.safetensors"
        result = minio_client.minio_upload_file(
            content=byte_buffer,
            s3_key=s3_key
        )

        t2 = time.time()
        # os.remove(model_path)
        show_log(f"Time upload to storage {t2-t1}")
        show_log(f"Result URL: {result}")

        # 4. Update task status
        is_success, response, error = update_status_for_task(
            UpdateStatusTaskRequest(
                task_id=request_data['task_id'],
                status="COMPLETED",
                result=result
            )
        )
        if not response:
            show_log(
                message="function: lora_trainer, "
                        f"celery_task_id: {celery_task_id}, "
                        f"error: {error}"
            )
            return False

        # 5. Send done task
        send_done_lora_trainner_task(
            DoneLoraTrainnerRequest(
                task_id=request_data['task_id'],
                url_download=result
            )
        )
        return True, response, None
    except Exception as e:
        print(str(e))
        return False, None, str(e)