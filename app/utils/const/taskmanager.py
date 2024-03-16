from tasks import (
    parrot_sd_task, 
    parrot_sdxl_task, 
    parrot_sdxl_lightning_task, 
    parrot_lora_trainer_task, 
    parrot_txt2vid_damo_task, 
    parrot_llm_gemma_7b_task, 
    parrot_t2s_task, 
    parrot_musicgen_task, 
    parrot_audiogen_task, 
    parrot_gte_task
)

TASK_MANAGEMENT = {
    "parrot_sd_task": {
        "queue_name": "sd_queue", 
        "task_name": "parrot_sd_task",
        "load_model_function": parrot_sd_task
    },
    "parrot_sdxl_task": {
        "queue_name": "sdxl_queue", 
        "task_name": "parrot_sdxl_task",
        "load_model_function": parrot_sdxl_task
    },
    "parrot_sdxl_lightning_task": {
        "queue_name": "sdxl_lightning_queue", 
        "task_name": "parrot_sdxl_lightning_task",
        "load_model_function": parrot_sdxl_lightning_task
    },
    "parrot_lora_trainer_task": {
        "queue_name": "lora_trainer_queue", 
        "task_name": "parrot_lora_trainer_task",
        "load_model_function": parrot_lora_trainer_task
    },
    "parrot_txt2vid_damo_task": {
        "queue_name": "txt2vid_damo_queue", 
        "task_name": "parrot_txt2vid_damo_task",
        "load_model_function": parrot_txt2vid_damo_task
    },
    "parrot_llm_gemma_7b_task": {
        "queue_name": "llm_gemma_7b_queue", 
        "task_name": "parrot_llm_gemma_7b_task",
        "load_model_function": parrot_llm_gemma_7b_task
    },
    "parrot_t2s_task": {
        "queue_name": "t2s_queue", 
        "task_name": "parrot_t2s_task",
        "load_model_function": parrot_t2s_task
    },
    "parrot_musicgen_task": {
        "queue_name": "musicgen_queue", 
        "task_name": "parrot_musicgen_task",
        "load_model_function": parrot_musicgen_task
    },
    "parrot_audiogen_task": {
        "queue_name": "audiogen_queue", 
        "task_name": "parrot_audiogen_task",
        "load_model_function": parrot_audiogen_task
    }, 
    "parrot_gte_task": {
        "queue_name": "gte_queue",
        "task_name": "parrot_gte_task",
        "load_model_function": parrot_gte_task
    }
}