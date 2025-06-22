import json
from pydantic import BaseModel, Field
from typing import Optional

class ModelConfig(BaseModel):
    tokenizer_pretrained_model: str = Field(default="t5-base", description="Pretrained model name for the tokenizer")
    max_input_length: Optional[int] = Field(..., description="Maximum token length for input question + context")
    batch_size: int = Field(..., description="Batch size required for Dataloader")
    model_name: str = Field(default="t5-base", description="Name of the model to use")
    learning_rate: float = Field(default=1e-4, description="Learning rate for the optimizer")
    dropout_rate: Optional[float] = Field(None, ge=0.0, le=1.0, description="Dropout rate to prevent overfitting")

class TrainingConfig(BaseModel):
    training_data_path: str = Field(..., description="Path to load the training data file")
    validation_data_path: str = Field(..., description="Path to load the validation data file")
    model: ModelConfig = Field(..., description="Model-related configuration")
    n_epochs: int = Field(..., description="Number of epochs for training the model")
    train_dir: str = Field(..., description="Directory to save the training files")
    clean_train_dir_before_training: bool = Field(default=True, description="Whether to clean the training directory before training")
    best_model_path: str = Field(..., description="Path to save the best model during training")
    losses_curve_path: str = Field(..., description="Path to save the losses curve during training")

def training_config_loader(config_path: str) -> TrainingConfig:
    try:
        with open(config_path, "r", encoding="utf-8") as file:
            config = json.load(file)
        return TrainingConfig(**config)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find training config file: {config_path}")