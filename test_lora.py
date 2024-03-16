from diffusers import AutoPipelineForText2Image
import torch

# pipeline = AutoPipelineForText2Image.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
pipeline = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16).to("cuda")
pipeline.load_lora_weights("/workspace/parrot-host/app/services/ai_services/lora_trainer/tmp/63662c50-ba65-4eca-b328-f714bd68db32/63662c50-ba65-4eca-b328-f714bd68db32.safetensors", weight_name="xyz.safetensors")
image = pipeline("xyz a handsome boy").images[0]

print(type(image))

image.save("test.png")