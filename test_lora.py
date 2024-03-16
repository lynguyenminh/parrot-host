from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
pipeline.load_lora_weights("/workspace/parrot-host/app/services/ai_services/lora_trainer/tmp/6bd791c2-e49d-4810-9aa0-58c5b5f97f96/6bd791c2-e49d-4810-9aa0-58c5b5f97f96.safetensors", weight_name="xyz.safetensors")
image = pipeline("a handsome boy").images[0]

print(type(image))

image.save("test.png")