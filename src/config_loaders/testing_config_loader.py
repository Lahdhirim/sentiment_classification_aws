import json
from pydantic import BaseModel, Field


class TestingConfig(BaseModel):
    test_data_path: str = Field(..., description="Path to load the test data file")
    trained_model_path: str = Field(..., description="Path to load the trained model")
    batch_size: int = Field(default=32, description="Batch size required for Dataloader")

def testing_config_loader(config_path: str) -> TestingConfig:
    try:
        with open(config_path, "r", encoding="utf-8") as file:
            config = json.load(file)
        return TestingConfig(**config)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find testing config file: {config_path}")