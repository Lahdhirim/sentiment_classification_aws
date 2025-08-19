import json
from pydantic import BaseModel, Field

class InferenceConfig(BaseModel):
    bucket_name: str = Field(..., description="Name of the S3 bucket containing the model")
    local_model_dir: str = Field(..., description="Local path where the model will be downloaded")
    s3_model_prefix: str = Field(..., description="S3 prefix (folder path) where model files are located")

def inference_config_loader(config_path: str) -> InferenceConfig:
    try:
        with open(config_path, "r", encoding="utf-8") as file:
            config = json.load(file)
        return InferenceConfig(**config)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find inference config file: {config_path}")