import os 

root_dir = os.path.dirname(os.path.abspath(__file__))
print(root_dir)

def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

lora_dir = os.path.join(root_dir, 'LoRA')

deps_dir = os.path.join(root_dir, "deps")

repo_dir = os.path.join(root_dir, "LoraTrainer")
training_dir = os.path.join(root_dir, "LoRA")

pretrained_model = os.path.join(root_dir, "pretrained_model")

vae_dir = os.path.join(root_dir, "vae")
config_dir = os.path.join(training_dir, "config")

accelerate_config = os.path.join(repo_dir, "accelerate_config/config.yaml")
tools_dir = os.path.join(repo_dir, "tools")
finetune_dir = os.path.join(repo_dir, "finetune")


# train_data_dir =  os.path.join(root_dir, "train_image_data")
# reg_data_dir = train_data_dir

create_dir(lora_dir)
create_dir(deps_dir)
create_dir(training_dir)
create_dir(pretrained_model)
create_dir(vae_dir)
create_dir(config_dir)
# create_dir(accelerate_config)
create_dir(pretrained_model)
create_dir(tools_dir)
create_dir(finetune_dir)
# create_dir(train_data_dir)




supported_types = [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG", ".webp", ".bmp"]


batch_size = 8
max_data_loader_n_workers = 2
model = "SmilingWolf/wd-v1-4-swinv2-tagger-v2"
threshold = 0.85


extension = ".txt"
append = False


# Model configuration
v_parameterization = False
project_name = "bnx"
v2 = False

pretrained_model_name_or_path = os.path.join(pretrained_model, "models--runwayml--stable-diffusion-v1-5/snapshots/1d0c4ebf6ff58a5caecab40fa1406526bca4b5b9")
pretrained_modelxl_name_or_path = os.path.join(pretrained_model, "models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b")
vae = os.path.join(root_dir, "pretrained_model/models--runwayml--stable-diffusion-v1-5/snapshots/1d0c4ebf6ff58a5caecab40fa1406526bca4b5b9/vae")
vae_xl = os.path.join(root_dir, "pretrained_model/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b/vae_1_0")

# output_dir = os.path.join(root_dir, "lora_output") #f"{root_dir}/SD-Data/Lora/"
output_to_drive = False
logging_dir = f"{root_dir}/LoRA/logs"



# create_dir(pretrained_model_name_or_path)

# create_dir(vae)

# create_dir(output_dir)
create_dir(logging_dir)


# Training configuration
train_repeats = 48
reg_repeats = 1
instance_token = "mksks"
class_token = "style"
resolution = 512
Batch_size = 6
flip_aug = False
caption_extension = ".txt"
caption_dropout_rate = 0
caption_dropout_every_n_epochs = 0
keep_tokens = 0

# Sampling configuration
enable_sample = True
sample_every_n_type = "sample_every_n_epochs"
sample_every_n_type_value = 2
sampler = "euler_a"
prompt = "portrait, masterpiece, ultra realistic,32k,extremely detailed CG unity 8k wallpaper, best quality"
negative = "watermark,text, logo,contact, error, blurry, cropped, username, artist name, (worst quality, low quality:1.4),"
width = 512
height = 768
scale = 7
seed = -1
steps = 20

# Network & Optimizer Config
network_category = "LoRA"
conv_dim = 1
conv_alpha = 1
network_dim = 32
network_alpha = 32
network_weight = ""
network_module = "lycoris.kohya" if network_category in ["LoHa", "LoCon_Lycoris"] else "networks.lora"
network_args = ""
optimizer_type = "Adafactor"
optimizer_args = ""
train_unet = True
unet_lr = 1e-4
train_text_encoder = True
text_encoder_lr = 5e-5
lr_scheduler = "constant"
lr_warmup_steps = 0
lr_scheduler_num_cycles = 0
lr_scheduler_power = 0

# Miscellaneous configuration
lowram = True
noise_offset = 0.0
num_epochs = 12
train_batch_size = Batch_size
mixed_precision = "fp16"
save_precision = "fp16"
save_n_epochs_type = "save_every_n_epochs"
save_n_epochs_type_value = 2
save_model_as = "safetensors"
max_token_length = 225
clip_skip = 1
gradient_checkpointing = False
gradient_accumulation_steps = 1
seed = -1

prior_loss_weight = 1.0
