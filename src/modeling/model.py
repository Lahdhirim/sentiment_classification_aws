from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from colorama import Fore, Style

class ModelBuilder:
    def __init__(
        self,
        model_name: str,
        tokenizer_pretrained_model: str,
        learning_rate: float
    ):
        self.model_name=model_name
        self.tokenizer_pretrained_model=tokenizer_pretrained_model
        self.learning_rate=learning_rate
  
    def _print_trainable_parameters(self, model: nn.Module) -> None:
        print(f"{Fore.CYAN}Trainable parameters:{Style.RESET_ALL}")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f" - {name}")

    def build_model(self) -> nn.Module:
        # Load the model
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        )
        print(f"{Fore.CYAN}Model loaded.{Style.RESET_ALL}")

        self._print_trainable_parameters(model)

        return model

    def build_tokenizer(self) -> AutoTokenizer:
        return AutoTokenizer.from_pretrained(self.tokenizer_pretrained_model, use_fast=True)
    
    def initialize(self):
        model = self.build_model()
        tokenizer = self.build_tokenizer()
        return model, tokenizer