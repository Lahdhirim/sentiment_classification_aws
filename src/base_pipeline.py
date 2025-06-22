from src.config_loaders.preprocessing_config_loader import PreprocessingConfig
from src.config_loaders.training_config_loader import TrainingConfig
from src.config_loaders.testing_config_loader import TestingConfig
from abc import ABC, abstractmethod
from typing import Union

class BasePipeline(ABC):
    """Abstract base class for main pipelines"""
    
    def __init__(self, config: Union[PreprocessingConfig, TrainingConfig, TestingConfig]):
        self.config = config
    
    @abstractmethod
    def run(self) -> None:
        """run the pipeline."""
        pass