import json
from pydantic import BaseModel, Field

class PreprocessingConfig(BaseModel):
    url_data: str = Field(..., description="URL of the input data")
    label_mapping_dict: dict[str, int] = Field(..., description="Mapping of labels to numerical values")
    test_size: float = Field(default=0.2, description="Proportion of the dataset to include in the test split")
    validation_size: float = Field(default=0.2, description="Proportion of the dataset to include in the validation split")
    training_data_path: str = Field(..., description="Path to save the training data file")
    validation_data_path: str = Field(..., description="Path to save the validation data file")
    test_data_path: str = Field(..., description="Path to save the test data file")

def preprocessing_config_loader(config_path: str) -> PreprocessingConfig:
    try:
        with open(config_path, "r", encoding="utf-8") as file:
            config = json.load(file)
        return PreprocessingConfig(**config)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find preprocessing config file: {config_path}")