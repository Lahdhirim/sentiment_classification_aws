from src.config_loaders.training_config_loader import TrainingConfig
from colorama import Fore, Style
from src.base_pipeline import BasePipeline
from src.utils.toolbox import load_csv_data, plot_training_and_validation_curves, clean_checkpoints
from datasets import Dataset
from src.modeling.model import ModelBuilder
from src.utils.schema import DataSchema, MetricSchema
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
        num_labels = train_data[DataSchema.LABEL].nunique() 
        label2id = {label: idx for idx, label in enumerate(sorted(train_data[DataSchema.SENTIMENT].unique()))}
        id2label = {idx: label for label, idx in label2id.items()}
        model_builder = ModelBuilder(model_name=self.config.model.model_name,
                                     num_labels=num_labels,
                                     id2label=id2label,
                                     label2id=label2id,
                                     tokenizer_pretrained_model=self.config.model.tokenizer_pretrained_model,
                                     learning_rate=self.config.model.learning_rate,
                                     freeze_backbone=self.config.model.freeze_backbone,
                                     dropout_rate=self.config.model.dropout_rate)
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
        print(f"{Fore.YELLOW}Starting the training loop...{Style.RESET_ALL}")
        args = TrainingArguments(
                output_dir=self.config.train_dir,
                overwrite_output_dir=True,
                num_train_epochs=self.config.n_epochs,
                learning_rate=self.config.model.learning_rate,
                lr_scheduler_type='constant', # Disable learning rate warmup (can result to a fast overfitting)
                per_device_train_batch_size=self.config.model.batch_size,
                per_device_eval_batch_size=self.config.model.batch_size,
                eval_strategy='epoch',
                logging_strategy='epoch',
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model=MetricSchema.ACCURACY,
                greater_is_better=True 
            )
        
        trainer = Trainer(
                model=model,
                args=args,
                train_dataset=train_dataset,
                eval_dataset=validation_dataset,
                compute_metrics=compute_accuracy,
                tokenizer=tokenizer
            )
        
        if self.config.clean_train_dir_before_training:
            clean_checkpoints(train_dir=self.config.train_dir)

        trainer.train()

        # Save the training and validation curves (loss and accuracy)
        training_logs = trainer.state.log_history
        train_losses = [log["loss"] for log in training_logs if "loss" in log]
        val_losses = [log["eval_loss"] for log in training_logs if "eval_loss" in log]
        val_accuracies = [log[f"eval_{MetricSchema.ACCURACY}"] for log in training_logs if f"eval_{MetricSchema.ACCURACY}" in log]
        plot_training_and_validation_curves(train_losses=train_losses,
                                            val_losses=val_losses,
                                            val_metrics=val_accuracies,
                                            save_path=self.config.training_curve_path)
        
        # Save the best model for Testing and Inference
        trainer.save_model(self.config.best_model_path)

        # Push the model to S3 bucket if enabled
        
        print(f"{Fore.GREEN}Training pipeline completed successfully!{Style.RESET_ALL}")

