import os
import torch
from diffusers import StableDiffusionPipeline


model_normal = "runwayml/stable-diffusion-v1-5" 
model_xl = "stabilityai/stable-diffusion-xl-base-1.0"

# check path is exist
path_save_model = './app/services/ai_services/lora_trainer/pretrained_model'
if os.path.exists(path_save_model) is False:
    os.makedirs(path_save_model)
    print(f"Create folder: {path_save_model}")
    # download the model
    pipe = StableDiffusionPipeline.from_pretrained(model_normal, torch_dtype=torch.float16, cache_dir=path_save_model)
    pipe = StableDiffusionPipeline.from_pretrained(model_xl, torch_dtype=torch.float16, cache_dir=path_save_model)
    print(f"[INFO] Download model to {path_save_model} successfully")
else: 
    list_item = os.listdir(path_save_model)
    if len(list_item) == 0:
        # download the model
        pipe = StableDiffusionPipeline.from_pretrained(model_normal, torch_dtype=torch.float16, cache_dir=path_save_model)
        pipe = StableDiffusionPipeline.from_pretrained(model_xl, torch_dtype=torch.float16, cache_dir=path_save_model)
        print(f"[INFO] Download model to {path_save_model} successfully")


