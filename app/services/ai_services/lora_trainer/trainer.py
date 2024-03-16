# add path to sys
import sys
sys.path.append('./app/services/ai_services/lora_trainer')

import subprocess
import os
import toml
import argparse
import config as cfg
import shutil


class LoraTraner: 
    @staticmethod
    def remove_files(dir_path):
        if os.path.isfile(dir_path):
            os.remove(dir_path)
        elif os.path.isdir(dir_path):
            shutil.rmtree(dir_path)
        else:
            print(f"Item '{dir_path}' is neither a file nor a directory.")

    @staticmethod
    def get_config_1(train_data_dir: str, reg_data_dir: str) -> dict: 
        config = {
            "general": {
                "enable_bucket": True,
                "caption_extension": cfg.caption_extension,
                "shuffle_caption": True,
                "keep_tokens": cfg.keep_tokens,
                "bucket_reso_steps": 64,
                "bucket_no_upscale": False,
            },
            "datasets": [
                {
                    "resolution": cfg.resolution,
                    "min_bucket_reso": 320 if cfg.resolution > 640 else 256,
                    "max_bucket_reso": 1280 if cfg.resolution > 640 else 1024,
                    "caption_dropout_rate": cfg.caption_dropout_rate if cfg.caption_extension == ".caption" else 0,
                    "caption_tag_dropout_rate": cfg.caption_dropout_rate if cfg.caption_extension == ".txt" else 0,
                    "caption_dropout_every_n_epochs": cfg.caption_dropout_every_n_epochs,
                    "flip_aug": cfg.flip_aug,
                    "color_aug": False,
                    "face_crop_aug_range": None,
                    "subsets": [
                        {
                            "image_dir": train_data_dir,
                            "class_tokens": f"{cfg.instance_token} {cfg.class_token}",
                            "num_repeats": cfg.train_repeats,
                        },
                        {
                            "is_reg": True,
                            "image_dir": reg_data_dir,
                            "class_tokens": cfg.class_token,
                            "num_repeats": cfg.reg_repeats,
                        }
                    ]
                }
            ]
        }
        return config

    @staticmethod
    def get_config_2(sdxl: str, user_name: str, output_dir: str, sample_every_n_type_value) -> dict:
        config = {
            "model_arguments": {
                "v2": cfg.v2,
                "v_parameterization": cfg.v_parameterization if cfg.v2 and cfg.v_parameterization else False,
                "pretrained_model_name_or_path": cfg.pretrained_model_name_or_path if sdxl == "0" else cfg.pretrained_modelxl_name_or_path,
                "vae": cfg.vae,
            },
            "additional_network_arguments": {
                "no_metadata": False,
                "unet_lr": float(cfg.unet_lr) if cfg.train_unet else None,
                "text_encoder_lr": float(cfg.text_encoder_lr) if cfg.train_text_encoder else None,
                "network_weights": cfg.network_weight,
                "network_module": cfg.network_module,
                "network_dim": cfg.network_dim,
                "network_alpha": cfg.network_alpha,
                "network_args": cfg.network_args,
                "network_train_unet_only": True if cfg.train_unet and not cfg.train_text_encoder else False,
                "network_train_text_encoder_only": True if cfg.train_text_encoder and not cfg.train_unet else False,
                "training_comment": None,
            },
            "optimizer_arguments": {
                "optimizer_type": cfg.optimizer_type,
                "learning_rate": cfg.unet_lr,
                "max_grad_norm": 1.0,
                "optimizer_args": eval(cfg.optimizer_args) if cfg.optimizer_args else None,
                "lr_scheduler": cfg.lr_scheduler,
                "lr_warmup_steps": cfg.lr_warmup_steps,
                "lr_scheduler_num_cycles": cfg.lr_scheduler_num_cycles if cfg.lr_scheduler == "cosine_with_restarts" else None,
                "lr_scheduler_power": cfg.lr_scheduler_power if cfg.lr_scheduler == "polynomial" else None,
            },
            "dataset_arguments": {
                "cache_latents": True,
                "debug_dataset": False,
            },
            "training_arguments": {
                "output_dir": output_dir,
                "output_name": user_name,
                "save_precision": cfg.save_precision,
                "save_every_n_epochs": cfg.save_n_epochs_type_value if cfg.save_n_epochs_type == "save_every_n_epochs" else None,
                "save_n_epoch_ratio": cfg.save_n_epochs_type_value if cfg.save_n_epochs_type == "save_n_epoch_ratio" else None,
                "save_last_n_epochs": None,
                "save_state": None,
                "save_last_n_epochs_state": None,
                "resume": None,
                "train_batch_size": cfg.train_batch_size,
                "max_token_length": 225,
                "mem_eff_attn": False,
                "xformers": True,
                "max_train_epochs": cfg.num_epochs,
                "max_data_loader_n_workers": 8,
                "persistent_data_loader_workers": True,
                "seed": cfg.seed if cfg.seed > 0 else None,
                "gradient_checkpointing": cfg.gradient_checkpointing,
                "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
                "mixed_precision": cfg.mixed_precision,
                "clip_skip": cfg.clip_skip if not cfg.v2 else None,
                "logging_dir": cfg.logging_dir,
                "log_prefix": user_name,
                "noise_offset": cfg.noise_offset if cfg.noise_offset > 0 else None,
                "lowram": cfg.lowram,
                "no_half_vae": True if sdxl == "1" else False
            },
            "sample_prompt_arguments":{
                "sample_every_n_steps": sample_every_n_type_value if cfg.sample_every_n_type == "sample_every_n_steps" else None,
                "sample_every_n_epochs": sample_every_n_type_value if cfg.sample_every_n_type == "sample_every_n_epochs" else None,
                "sample_sampler": cfg.sampler,
            },
            "dreambooth_arguments":{
                "prior_loss_weight": 1.0,
            },
            "saving_arguments":{
                "save_model_as": cfg.save_model_as
            },
        }

        return config

    @staticmethod
    def create_sample_prompt() -> None: 
        try: 
            # if is_male == "1": 
            #     sample_str = f"""1man, {cfg.prompt} --n {cfg.negative} --w {cfg.width} --h {cfg.height} --l {cfg.scale} --s {cfg.steps} {'--d ' + str(cfg.seed) if cfg.seed > 0 else ''}"""
            # elif is_male == "0": 
            #     sample_str = f"""1girl, {cfg.prompt} --n {cfg.negative} --w {cfg.width} --h {cfg.height} --l {cfg.scale} --s {cfg.steps} {'--d ' + str(cfg.seed) if cfg.seed > 0 else ''}"""

            sample_str = f"""{cfg.prompt} --n {cfg.negative} --w {cfg.width} --h {cfg.height} --l {cfg.scale} --s {cfg.steps} {'--d ' + str(cfg.seed) if cfg.seed > 0 else ''}"""

            prompt_path = os.path.join(cfg.config_dir, "sample_prompt.txt")
            
            with open(prompt_path, "w") as f:
                f.write(sample_str)

            print("[INFO] Sample prompt file created successfully. Save at ", prompt_path)
        except Exception as e:
            print(f"Failed to create sample prompt file. Error: {str(e)}")

    @staticmethod
    def write_config(config: dict, config_path: str) -> None:
        try: 
            with open(config_path, "w") as f:
                f.write(config)
            print(f"[INFO] Config file written successfully. Save at {config_path}")
        except Exception as e:
            print(f"Failed to write config file. Error: {str(e)}")

    @staticmethod
    def transfer_json_to_toml(json_data: dict) -> str: 
        try: 
            toml_data = toml.dumps(json_data)
            return toml_data
        except Exception as e:
            print(f"Failed to convert json to toml. Error: {str(e)}")
            return None

    @staticmethod
    def replace_empty_with_none(config):
        for key in config:
            if isinstance(config[key], dict):
                for sub_key in config[key]:
                    if config[key][sub_key] == "":
                        config[key][sub_key] = None
            elif config[key] == "":
                config[key] = None
        return config

    @staticmethod
    def change_dir(dir_path: str) -> None:
        try: 
            os.chdir(dir_path)
            print(f"[INFO] Changed directory to {dir_path}")
        except Exception as e:
            print(f"Failed to change directory to {dir_path}. Error: {str(e)}")

    @staticmethod
    def train_script(sdxl: str):
        print(f"[INFO] Running train script with sdxl: {sdxl}") 
        print(f"Current directory: {os.getcwd()}")
        print(f"Config file: {cfg.accelerate_config}")
        print(f"Sample prompt: {os.path.join(cfg.root_dir, 'LoRA/config/sample_prompt.txt')}")
        print(f"Dataset config: {os.path.join(cfg.root_dir, 'LoRA/config/dataset_config.toml')}")
        print(f"Config file: {os.path.join(cfg.root_dir, 'LoRA/config/config_file.toml')}")
        try: 
            if sdxl == "1":
                subprocess.run([
                    "accelerate", "launch",
                    "--config_file", cfg.accelerate_config,
                    "--num_cpu_threads_per_process", "1",
                    "sdxl_train_network.py",
                    "--sample_prompts", os.path.join(cfg.root_dir, "LoRA/config/sample_prompt.txt"),
                    "--dataset_config", os.path.join(cfg.root_dir, "LoRA/config/dataset_config.toml"),
                    "--config_file", os.path.join(cfg.root_dir, "LoRA/config/config_file.toml")
                ])
                print("[INFO] Lora sdxl train script ran successfully.")
            else:
                subprocess.run([
                    "accelerate", "launch",
                    "--config_file", cfg.accelerate_config,
                    "--num_cpu_threads_per_process", "1",
                    "train_network.py",
                    "--sample_prompts", os.path.join(cfg.root_dir, "LoRA/config/sample_prompt.txt"),
                    "--dataset_config", os.path.join(cfg.root_dir, "LoRA/config/dataset_config.toml"),
                    "--config_file", os.path.join(cfg.root_dir, "LoRA/config/config_file.toml")
                ])
                print("[INFO] LoRA 1.5 train script ran successfully.")
        except Exception as e:
            print(f"Failed to run train script. Error: {str(e)}")
            
    @staticmethod
    def tag_image_script(train_data_dir: str) -> None:
        try: 
            subprocess.run([
                "python", "tag_images_by_wd14_tagger.py",
                train_data_dir,
                "--batch_size", "4",
                "--repo_id", cfg.model,
                "--thresh", str(cfg.threshold),
                "--caption_extension", ".txt",
                "--max_data_loader_n_workers", str(cfg.max_data_loader_n_workers)
            ])
            print("[INFO] Tag image script ran successfully.")
        except Exception as e:
            print(f"Failed to run tag image script. Error: {str(e)}")


    def run(self, train_data_dir, user_name, sdxl):
        try: 
            
            reg_data_dir = train_data_dir
                    
            self.change_dir(cfg.root_dir)
            for item in os.listdir(train_data_dir): #remove data that are not supported
                if os.path.splitext(item)[1] not in cfg.supported_types: 
                    self.remove_files(os.path.join(train_data_dir, item))

            self.change_dir(cfg.finetune_dir)
            self.tag_image_script(train_data_dir)
            print(f"[INFO] Removed unsupported files from {train_data_dir}")


        
            self.write_config(
                self.transfer_json_to_toml(
                   self.replace_empty_with_none(
                        self.get_config_1(train_data_dir, reg_data_dir)
                    )
                ), 
                os.path.join(cfg.config_dir, "dataset_config.toml")
            )


            sample_every_n_type_value = 2 if cfg.enable_sample else 999999
                

            self.create_sample_prompt()


            self.write_config(
                self.transfer_json_to_toml(
                    self.replace_empty_with_none(
                        self.get_config_2(sdxl, user_name, train_data_dir, sample_every_n_type_value)
                    )
                ), 
                os.path.join(cfg.config_dir, "config_file.toml")
            )
            
            
            self.change_dir(cfg.repo_dir)
            self.train_script(sdxl=sdxl)

            print("[INFO] Train script ran successfully.")
            print(train_data_dir)
            return os.path.join(train_data_dir, f"{user_name}.safetensors")
        except Exception as e:
            print(f"[INFO] Failed to run train script. Error: {str(e)}")
            return None


if __name__ ==  '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, default='', type=str)
    parser.add_argument("--user", required=True, default='', type=str)
    parser.add_argument("--sdxl", required=True, default='0', type=str)
    # parser.add_argument("--is_male", required=True, default="1", type=str)
    args = parser.parse_args()
    
   
    # test trainer
    lora = LoraTraner()
    # lora.run(args.data_dir, args.user, args.sdxl, args.is_male)
    
    