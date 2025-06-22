import json
from pydantic import BaseModel, Field
from typing import List, Optional

class PushCondition(BaseModel):
    metric: str = Field(..., description="Metric name to evaluate (e.g. accuracy, precision)")
    threshold: float = Field(..., description="Threshold value to trigger model push")

class PushModelS3Config(BaseModel):
    enabled: bool = Field(..., description="Whether to push model to S3")
    conditions: List[PushCondition] = Field(default_factory=list, description="Conditions to satisfy before pushing")
    bucket_name: str = Field(..., description="S3 bucket name")
    prefix: str = Field(..., description="Prefix path in the S3 bucket")


class TestingConfig(BaseModel):
    test_data_path: str = Field(..., description="Path to load the test data file")
    trained_model_path: str = Field(..., description="Path to load the trained model")
    batch_size: int = Field(default=32, description="Batch size required for Dataloader")
    metrics_output_file: str = Field(..., description="Path to save the performance metrics")
    push_model_s3: Optional[PushModelS3Config] = Field(None, description="Optional S3 push config")

def testing_config_loader(config_path: str) -> TestingConfig:
    try:
        with open(config_path, "r", encoding="utf-8") as file:
            config = json.load(file)
        return TestingConfig(**config)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find testing config file: {config_path}")