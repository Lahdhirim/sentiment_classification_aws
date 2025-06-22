from src.config_loaders.training_config_loader import TrainingConfig
from colorama import Fore, Style
from src.base_pipeline import BasePipeline
from src.utils.toolbox import load_csv_data
from datasets import Dataset
from src.modeling.model import ModelBuilder
from src.utils.schema import DataSchema
from transformers import TrainingArguments, Trainer
from src.evaluators.accuracy import compute_accuracy

class TrainingPipeline(BasePipeline):    
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
    
    def run(self):
        
        print(f"{Fore.GREEN}Starting training pipeline...{Style.RESET_ALL}")

        # Load the data
        print(f"{Fore.YELLOW}Loading data from specified paths...{Style.RESET_ALL}")
        train_data = load_csv_data(data_source=self.config.training_data_path)
        validation_data = load_csv_data(data_source=self.config.validation_data_path)

        # Create the dataset objects
        print(f"{Fore.YELLOW}Creating dataset objects...{Style.RESET_ALL}")
        train_dataset = Dataset.from_pandas(train_data)
        validation_dataset = Dataset.from_pandas(validation_data)

        # Create the model and the tokenizer
        print(f"{Fore.YELLOW}Creating model and tokenizer...{Style.RESET_ALL}")
        model_builder = ModelBuilder(model_name=self.config.model.model_name,
                                     tokenizer_pretrained_model=self.config.model.tokenizer_pretrained_model,
                                     learning_rate=self.config.model.learning_rate)
        model, tokenizer = model_builder.initialize()

        train_dataset = train_dataset.map(
                            lambda batch: tokenizer(batch[DataSchema.REVIEW], padding=True, truncation=True, max_length=self.config.model.max_input_length),
                            batched=True,
                            batch_size=None
                        )
        validation_dataset = validation_dataset.map(
                            lambda batch: tokenizer(batch[DataSchema.REVIEW], padding=True, truncation=True, max_length=self.config.model.max_input_length),
                            batched=True,
                            batch_size=None
                        )
        
        # Train the model
        args = TrainingArguments(
                output_dir=self.config.train_dir,
                overwrite_output_dir=True,
                num_train_epochs=self.config.n_epochs,
                learning_rate=self.config.model.learning_rate,
                per_device_train_batch_size=self.config.model.batch_size,
                per_device_eval_batch_size=self.config.model.batch_size,
                eval_strategy='epoch'
            )
        
        trainer = Trainer(
                model=model,
                args=args,
                train_dataset=train_dataset,
                eval_dataset=validation_dataset,
                compute_metrics=compute_accuracy,
                tokenizer=tokenizer
            )
        trainer.train()
        
        print(f"{Fore.GREEN}Training pipeline completed successfully!{Style.RESET_ALL}")

