from src.config_loaders.preprocessing_config_loader import PreprocessingConfig
from src.base_pipeline import BasePipeline
from colorama import Fore, Style
from src.utils.toolbox import load_csv_data
from src.utils.schema import DataSchema
from sklearn.model_selection import train_test_split 

class PreprocessingPipeline(BasePipeline):
    def __init__(self, config: PreprocessingConfig):
        super().__init__(config)
    
    def run(self):

        print(f"{Fore.GREEN}Starting preprocessing pipeline...{Style.RESET_ALL}")

        # Load the input data
        print(f"{Fore.YELLOW}Loading data from: {self.config.url_data}{Style.RESET_ALL}")
        data = load_csv_data(data_source=self.config.url_data)
        print(f"{Fore.CYAN}Data shape before preprocessing: {data.shape}{Style.RESET_ALL}")

        # Process the data
        print(f"{Fore.YELLOW}Cleaning data...{Style.RESET_ALL}")
        label_mapping_dict = self.config.label_mapping_dict
        assert set(label_mapping_dict.keys()) == set(data[DataSchema.SENTIMENT].unique()), f"Label mapping dict should match the unique sentiment values in the data: {data[DataSchema.SENTIMENT].unique()}. Got {list(label_mapping_dict.keys())} instead."
        data[DataSchema.LABEL] = data[DataSchema.SENTIMENT].map(label_mapping_dict).astype(int)

        # Split the data into training, validation, and test sets
        print(f"{Fore.YELLOW}Splitting data into training, validation, and test sets...{Style.RESET_ALL}")
        train_data, test_data = train_test_split(data, test_size=self.config.test_size, random_state=42)
        train_data, val_data = train_test_split(train_data, test_size=self.config.validation_size, random_state=42)
        print(f"{Fore.CYAN}Data shapes after splitting - Train: {train_data.shape}, Validation: {val_data.shape}, Test: {test_data.shape}{Style.RESET_ALL}")

        # Save the split data to the specified paths
        print(f"{Fore.YELLOW}Saving data to specified paths...{Style.RESET_ALL}")
        train_data.to_csv(self.config.training_data_path, index=False)
        val_data.to_csv(self.config.validation_data_path, index=False)
        test_data.to_csv(self.config.test_data_path, index=False)

        print(f"{Fore.GREEN}Preprocessing pipeline completed successfully!{Style.RESET_ALL}")

        
        