from typing import Union, Any, List, Dict

from pydantic import BaseModel, Field


class LoraTrainnerRequest(BaseModel):
    task_id: str = Field(..., description="task_id")
    prompt: list[str] = Field(..., description="prompt")
    minio_input_paths: list[str] = Field(..., description="minio_input_paths")
    is_sdxl: str = Field(..., description="1: SDXL, 0: SD1.5")


class DoneLoraTrainnerRequest(BaseModel):
    task_id: str = Field(..., description="task_id")
    url_download: Any = Field(..., description="url_download")


class UpdateStatusTaskRequest(BaseModel):
    task_id: str = Field(..., description="task_id")
    status: str = Field(..., description="status")
    result: Union[str, List, Dict, None] = Field(None, description="result")
    vector_result: Any = Field(None, description="vector_result")


class SendFailTaskRequest(BaseModel):
    task_id: str = Field(..., description="task_id")
    task_type: str = Field(..., description="task_type")
    sub_task_id: Union[str, None] = Field(None, description="sub_task_id")


class SendDoneTaskRequest(BaseModel):
    task_id: str = Field(..., description="task_id")
    task_type: str = Field(..., description="task_type")
    sub_task_id: Union[str, None] = Field(None, description="sub_task_id")


class SendProgressTaskRequest(BaseModel):
    task_id: str = Field(..., description="task_id")
    task_type: str = Field(..., description="task_type")
    percent: int = Field(..., description="percent")
    sub_task_id: Union[str, None] = Field(None, description="sub_task_id")


class SDXLRequest(BaseModel):
    task_id: str = Field(..., description="task_id")
    prompt: str = Field(..., description="prompt")
    config: dict = Field(..., description="config")


class DoneSDXLRequest(BaseModel):
    task_id: str = Field(..., description="task_id")
    url_download: Any = Field(..., description="url_download")


class SDRequest(BaseModel):
    task_id: str = Field(..., description="task_id")
    prompt: str = Field(..., description="prompt")
    config: dict = Field(..., description="config")


class DoneSDRequest(BaseModel):
    task_id: str = Field(..., description="task_id")
    url_download: Any = Field(..., description="url_download")


class LLMRequest(BaseModel):
    task_id: str = Field(..., description="task_id")
    messages: list = Field(..., description="messages")
    config: dict = Field(..., description="config")

class GTERequest(BaseModel):
    task_id: str = Field(..., description="task_id")
    prompt: str = Field(..., description="prompt")
    config: dict = Field(..., description="config")

class DoneLLMRequest(BaseModel):
    task_id: str = Field(..., description="task_id")
    response: Any = Field(..., description="url_download")

class DoneGTERequest(BaseModel):
    task_id: str = Field(..., description="task_id")
    response: Any = Field(..., description="response")

class T2SRequest(BaseModel):
    task_id: str = Field(..., description="task_id")
    prompt: str = Field(..., description="prompt")
    config: dict = Field(..., description="config")


class DoneT2SRequest(BaseModel):
    task_id: str = Field(..., description="task_id")
    url_download: Any = Field(..., description="url_download")

class MusicGenRequest(BaseModel):
    task_id: str = Field(..., description="task_id")
    prompt: str = Field(..., description="prompt")
    config: dict = Field(..., description="config")

class DoneMusicGenRequest(BaseModel):
    task_id: str = Field(..., description="task_id")
    url_download: Any = Field(..., description="url_download")

class AudioGenRequest(BaseModel):
    task_id: str = Field(..., description="task_id")
    prompt: str = Field(..., description="prompt")
    config: dict = Field(..., description="config")

class DoneAudioGenRequest(BaseModel):
    task_id: str = Field(..., description="task_id")
    url_download: Any = Field(..., description="url_download")