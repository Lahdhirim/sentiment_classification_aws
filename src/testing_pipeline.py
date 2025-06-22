from src.config_loaders.testing_config_loader import TestingConfig
from colorama import Fore, Style
from src.base_pipeline import BasePipeline
from src.utils.toolbox import load_csv_data
from src.utils.schema import MetricSchema
import torch
from src.evaluators.accuracy import compute_accuracy
from transformers import pipeline


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
            print(Fore.MAGENTA + f"Model loaded from {self.config.trained_model_path}." + Style.RESET_ALL)
        except FileNotFoundError:
            raise FileNotFoundError(
            Fore.RED + f"Could not find the model at {self.config.trained_model_path}. Please check the path and try again." + Style.RESET_ALL)

        data = ['this movie was horrible, the plot was really boring. acting was okay',
                'the movie is really sucked. there is not plot and acting was bad',
                'what a beautiful movie. great plot. acting was good. will see it again'
                ]
        
        print(f"{Fore.GREEN}Testing pipeline completed successfully!{Style.RESET_ALL}")
            