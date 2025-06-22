import argparse
import subprocess

# Import config loaders and pipeline classes for each task
from src.config_loaders.preprocessing_config_loader import preprocessing_config_loader
from src.preprocessing_pipeline import PreprocessingPipeline

from src.config_loaders.training_config_loader import training_config_loader
from src.training_pipeline import TrainingPipeline

from src.config_loaders.testing_config_loader import testing_config_loader
from src.testing_pipeline import TestingPipeline

if __name__ == "__main__":

    # Parse command-line argument to determine which mode to run
    parser = argparse.ArgumentParser(description="Sentiment Prediction")
    parser.add_argument("mode", choices=["process_data", "train", "test", "inference"],
                        default="process_data", nargs="?", help="Choose mode: process_data, train, test, or inference")
    args = parser.parse_args()

    # Launch the appropriate pipeline based on the selected mode

    if args.mode == "process_data":
        # Load processing config and run data preprocessing pipeline
        processing_config = preprocessing_config_loader(config_path="config/preprocessing_config.json")
        processing_pipeline = PreprocessingPipeline(config=processing_config)
        processing_pipeline.run()
    
    elif args.mode == "train":
        # Load training config and run training pipeline
        training_config = training_config_loader(config_path="config/training_config.json")
        training_pipeline = TrainingPipeline(config=training_config)
        training_pipeline.run()
    
    elif args.mode == "test":
        # Load testing config and run testing pipeline
        testing_config = testing_config_loader(config_path="config/testing_config.json")
        testing_pipeline = TestingPipeline(config=testing_config)
        testing_pipeline.run()
    
    elif args.mode == "inference":
        # Run the Streamlit app
        subprocess.run(["streamlit", "run", "src/web_app/app.py"])
    
    else:
        print("Invalid mode. Please choose 'process_data', 'train', 'test', or 'inference'.")