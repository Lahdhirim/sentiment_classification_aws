import streamlit as st
from transformers import pipeline
from src.aws_services.s3_service import S3Manager
from src.config_loaders.inference_config_loader import inference_config_loader

@st.cache_resource
def load_pipeline():
    """Download model from S3 and load the HuggingFace pipeline."""

    # Load configuration
    config = inference_config_loader(config_path="config/inference_config.json")
    s3_manager = S3Manager(bucket_name=config.bucket_name)
    s3_manager.download_directory(s3_prefix=config.s3_model_prefix, local_directory_path=config.local_model_dir)

    # Build pipeline
    return pipeline(config.model_task, model=config.local_model_dir)

# UI
st.title("Sentiment Analysis Application")
classifier = load_pipeline()

user_input = st.text_area("Enter your review:", "This movie was amazing!")
if st.button("Predict"):
    with st.spinner("Analyzing sentiment..."):
        prediction = classifier(user_input)
        score = round(prediction[0]['score'] * 100, 2)
        st.success(f"Prediction: {prediction[0]['label']} ({score}%)")  