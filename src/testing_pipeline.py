from src.config_loaders.testing_config_loader import TestingConfig
from colorama import Fore, Style
from src.base_pipeline import BasePipeline
from src.utils.toolbox import load_csv_data
from src.utils.schema import DataSchema
from transformers import pipeline
from src.evaluators.testing_metrics import MetricsCalculator

class TestingPipeline(BasePipeline):    
    def __init__(self, config: TestingConfig):
        super().__init__(config)
    
    def run(self):

        print(f"{Fore.GREEN}Starting testing pipeline...{Style.RESET_ALL}")

        # Load the data
        print(f"{Fore.YELLOW}Loading data from specified path...{Style.RESET_ALL}")
        test_data = load_csv_data(data_source=self.config.test_data_path)

        # Load the model
        print(f"{Fore.YELLOW}Loading trained model from {self.config.trained_model_path}{Style.RESET_ALL}")
        try:
            classifier = pipeline("text-classification", model=self.config.trained_model_path)
            print(Fore.MAGENTA + f"Model and Configuration loaded from {self.config.trained_model_path}." + Style.RESET_ALL)

        except FileNotFoundError:
            raise FileNotFoundError(Fore.RED + f"Could not find the model at {self.config.trained_model_path}. Please check the path and try again." + Style.RESET_ALL)
        
        # Make predictions on the test data
        predictions = classifier(test_data[DataSchema.REVIEW].tolist(), truncation=True, max_length=512)

        # Evaluate the model
        pred_labels = [pred[DataSchema.LABEL] for pred in predictions]
        true_labels = test_data[DataSchema.SENTIMENT].tolist()
        
        metrics_calculator = MetricsCalculator(true_labels=true_labels, 
                                               pred_labels=pred_labels,
                                               output_csv_path=self.config.metrics_output_file)
        metrics_calculator.calculate_metrics()
        
        print(f"{Fore.GREEN}Testing pipeline completed successfully!{Style.RESET_ALL}")
            