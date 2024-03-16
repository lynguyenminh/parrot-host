from app.src.v1.lora_trainer_v1.lora_trainer import lora_trainer



if __name__=="__main__": 
    result = lora_trainer(
        celery_task_id="123",
        request_data={
            "task_id": "123",
            "is_sdxl": "1", 
            "prompt": ["solo, shirt, 1boy, male_focus, realistic", "solo, shirt, 1boy, male_focus, realistic", "solo, shirt, 1boy, male_focus, realistic", "solo, shirt, 1boy, male_focus, realistic", "solo, shirt, 1boy, male_focus, realistic"],
            "minio_input_paths": ["https://i.pinimg.com/236x/c2/9a/7d/c29a7d29348b1a3f502803ab9d8355cc.jpg", 
                                  "https://ivcdn.vnecdn.net/giaitri/images/web/2019/07/01/mv-son-tung-m-tp-dat-47-trieu-luot-xem-sau-50-phut-ra-mat-1561993656.jpg", 
                                  "https://bcp.cdnchinhphu.vn/334894974524682240/2022/4/29/son-tung-mtp-08110778-1651217357607701319631.jpg", 
                                  "https://imagev3.vietnamplus.vn/w1000/Uploaded/2024/ymtih/2016_03_24/174602sontungmtp.jpg.webp", 
                                  "https://nld.mediacdn.vn/2019/12/18/son-tung-mtp-lan-mat-tam-hon-1-nam-2-thang-chuyen-gi-da-xay-ra-2-15766471244331034980603.jpg"]
        }        
    )

    print(result)