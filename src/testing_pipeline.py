from src.config_loaders.testing_config_loader import TestingConfig
from colorama import Fore, Style
from src.base_pipeline import BasePipeline
from src.utils.toolbox import load_csv_data
from src.utils.schema import DataSchema, MetricSchema
from transformers import pipeline
from src.evaluators.testing_metrics import MetricsCalculator
from src.aws_services.s3_service import S3Manager

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
        metrics = metrics_calculator.calculate_metrics()
        print(f"{Fore.CYAN}Model Evaluation Metrics: Accuracy: {metrics[MetricSchema.ACCURACY]:.4f}, Precision: {metrics[MetricSchema.PRECISION]:.4f}, Recall: {metrics[MetricSchema.RECALL]:.4f}, F1 Score: {metrics[MetricSchema.F1_SCORE]:.4f}{Style.RESET_ALL}")

        # Push the model to S3 Bucket if it reaches the required performances
        if self.config.push_model_s3:
            push_model_s3_config = self.config.push_model_s3
            if push_model_s3_config.enabled:
                if len(push_model_s3_config.conditions)>0:
                    print(f"{Fore.YELLOW}Verifying conditions to push model to S3 bucket...{Style.RESET_ALL}")
                    model_valid = True
                    for condition in push_model_s3_config.conditions:
                        assert condition.metric in metrics, f"Metric {condition.metric} not found in the metrics dictionary."
                        if metrics[condition.metric] < condition.threshold :
                            model_valid = False
                            print(f"{Fore.RED}Metric {condition.metric} ({metrics[condition.metric]}) does not meet the threshold of {condition.threshold}. Model will not be pushed to S3 bucket...{Style.RESET_ALL}")
                            break
                    
                    if model_valid:
                        print(f"{Fore.GREEN}Model validation passed. Pushing model to S3 bucket...{Style.RESET_ALL}")
                        s3_manager = S3Manager(bucket_name=push_model_s3_config.bucket_name)
                        s3_manager.create_bucket_if_not_exists()
                        s3_manager.upload_directory(local_directory_path=self.config.trained_model_path, 
                                                    s3_prefix=push_model_s3_config.prefix)

                else:
                    print(f"{Fore.RED}No conditions specified for pushing model to S3 bucket. Skipping the push operation...{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}Pushing model to S3 bucket is disabled. Skipping the push operation...{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}Pushing model to S3 bucket is not configured. Skipping the push operation...{Style.RESET_ALL}")

        
        print(f"{Fore.GREEN}Testing pipeline completed successfully!{Style.RESET_ALL}")
            