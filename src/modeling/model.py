from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from colorama import Fore, Style
from typing import Optional

class ModelBuilder:
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        id2label: dict,
        label2id: dict,
        tokenizer_pretrained_model: str,
        learning_rate: float,
        dropout_rate: Optional[float] = None,
    ):
        self.model_name=model_name
        self.num_labels=num_labels
        self.id2label=id2label
        self.label2id=label2id
        self.tokenizer_pretrained_model=tokenizer_pretrained_model
        self.learning_rate=learning_rate
        self.dropout_rate=dropout_rate
  
    def _print_trainable_parameters(self, model: nn.Module) -> None:
        print(f"{Fore.CYAN}Trainable parameters:{Style.RESET_ALL}")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f" - {name}")

    def build_model(self) -> nn.Module:
        # Load the model
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id
        )
        print(f"{Fore.CYAN}Model loaded.{Style.RESET_ALL}")

        # Control Dropout rate
        if self.dropout_rate:
            if hasattr(model, "dropout"):
                model.dropout.p=self.dropout_rate
                print(f"{Fore.CYAN}Dropout rate set to {self.dropout_rate}.{Style.RESET_ALL}")
            else:
                print(f"{Fore.CYAN}Model does not have a dropout layer.{Style.RESET_ALL}")
        else:
            print(f"{Fore.CYAN}No dropout rate specified.{Style.RESET_ALL}")

        # Print trainable parameters
        self._print_trainable_parameters(model)

        return model

    def build_tokenizer(self) -> AutoTokenizer:
        return AutoTokenizer.from_pretrained(self.tokenizer_pretrained_model, use_fast=True)
    
    def initialize(self):
        model = self.build_model()
        tokenizer = self.build_tokenizer()
        return model, tokenizer