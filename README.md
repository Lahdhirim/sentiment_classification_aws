# Sentiment Classifier Applicaton using **Hugging Face Transformers** and **AWS Services (S3, EC2)**

This project is a sentiment classifier tool built using **Hugging Face** libraries and deployed using **AWS services** such as S3 for data storage, EC2 for running the tool during inference. It provides a fully customizable pipelines for:
- Data processing
- Model fine-tuning
- Model evaluation including automatic pushing to S3 bucket based on performance metrics\

The final model is deployed on both **Streamlit Share Server** (Public link: https://sentimentclassificationaws-nq67u8bs4qsjdwcqsa4lso.streamlit.app/) and **AWS EC2** instance.

![Demo](assets/demo_streamlit.gif)

## Output Examples
- Review: I recently watched this movie and was absolutely blown away by the performances. The story was heartwarming and the characters were incredibly well-developed. It kept me engaged from beginning to end, and I couldn’t stop smiling by the time it was over. Definitely one of the best films I’ve seen this year! ⇒ **Prediction: positive (98.64%)**
- Review: I contacted customer support for help with my account, but the experience was terrible. The representative was rude, unhelpful, and clearly not interested in resolving my issue. After waiting on hold for nearly an hour, I was told to check the FAQ instead. I left feeling frustrated and ignored ⇒ **Prediction: negative (98.61%)**

## Applications
This tool can be used in various applications such as:
- Sentiment analysis for social media posts (e.g., Twitter, Facebook)
- Sentiment analysis for customer reviews
- Analyzing feedback from surveys and questionnaires
- Enhancing chatbot or virtual assistant responses based on sentiment
- Assessing employee sentiment from internal communication
- Real-time analysis of financial news and investor sentiment

## Pipelines Overview

### Data Preprocessing ([src/preprocessing_pipeline.py](src/preprocessing_pipeline.py))
*Configurable via:* [config/preprocessing_config.json](config/preprocessing_config.json)
| Parameter              | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| `url_data`             | URL of the raw dataset to download (in CSV format)                         |
| `label_mapping_dict`   | Dictionary mapping text labels (e.g., "positive") to numerical values (e.g., 1) |
| `test_size`            | Proportion of the dataset to be used as the test set (e.g., 0.2 = 20%)     |
| `validation_size`      | Proportion of the training set to be used for validation                   |
| `training_data_path`   | File path to save the processed training dataset                           |
| `validation_data_path` | File path to save the processed validation dataset                         |
| `test_data_path`       | File path to save the processed test dataset                              |

The main steps of the data preprocessing pipeline are as follows:
- Load data and map labels to numerical values. 
- Split into training, validation, and test sets.  
- Save processed datasets for later stages.

### Training Pipeline ([src/training_pipeline.py](src/training_pipeline.py)) 
*Configurable via:* [config/training_config.json](config/training_config.json)
| Parameter                          | Description                                                                                   |
|------------------------------------|-----------------------------------------------------------------------------------------------|
| `training_data_path`               | Path to the CSV file containing training data                                                |
| `validation_data_path`             | Path to the CSV file containing validation data                                              |
| `model.tokenizer_pretrained_model`| Name of the pretrained model used to tokenize input text                                     |
| `model.max_input_length`           | Maximum length (in tokens) for each input sequence                                           |
| `model.batch_size`                 | Number of samples processed before the model is updated                                      |
| `model.model_name`                 | Name or path of the model architecture to use                                                |
| `model.learning_rate`              | Learning rate used by the optimizer during training                                          |
| `model.dropout_rate`               | Proportion of units randomly dropped to prevent overfitting (between 0 and 1)                |
| `model.freeze_backbone`            | Whether to freeze the model's backbone layers during training (useful to speed up training)  |
| `n_epochs`                         | Number of complete passes through the training dataset                                       |
| `train_dir`                        | Directory to save the training files                            |
| `clean_train_dir_before_training`  | If `true`, deletes the content of `train_dir` before starting training                       |
| `best_model_path`                  | File path where the best model (based on validation performance) will be saved               |
| `training_curve_path`              | File path to save the training/validation loss and metrics plots  

The main steps of the train pipeline are as follows:
- Load configuration and initialize model and tokenizer. 
- Fine-tune the model on training data. using the library `transformers` from Hugging Face. 
- Save the best model based on validation loss.
- Save training curves to visualize performance during training.

### Evaluation Pipeline ([src/testing_pipeline.py](src/testing_pipeline.py))
*Configurable via:* [config/testing_config.json](config/testing_config.json)
| Parameter                              | Description                                                                                     |
|----------------------------------------|-------------------------------------------------------------------------------------------------|
| `test_data_path`                       | Path to the CSV file containing test data                                                      |
| `trained_model_path`                   | Path to the trained model to be loaded for testing                                             |
| `batch_size`                           | Number of test samples processed at once during evaluation                                     |
| `metrics_output_file`                  | Path to save the calculated evaluation metrics (e.g., accuracy, precision, recall)            |
| `push_model_s3.enabled`                | If `true`, allows pushing the model to an S3 bucket if defined conditions are met              |
| `push_model_s3.conditions`             | List of metric-based conditions that must be satisfied to trigger a model push                 |
| `push_model_s3.conditions.metric`    | Name of the metric to check (e.g., `accuracy`, `precision`)                                    |
| `push_model_s3.conditions.threshold` | Minimum required value for the metric to allow model upload to S3                              |
| `push_model_s3.bucket_name`            | Name of the S3 bucket where the model will be uploaded                                         |
| `push_model_s3.prefix`                 | Folder or path prefix in the bucket under which the model will be stored                      |

The main steps of the evaluation pipeline are as follows:
- Load the best trained model.  
- Predict on the test set.  
- Evaluate the model using classification metrics (e.g., accuracy, precision, recall).
- Save the performance metrics to a CSV file.
- Push the model to S3 if defined conditions are satisfied.

### Inference via Web Application

- Allows users to input a review and receive a sentiment prediction.

## AWS Services Configuration for Model Deployment
1. Create a User using **AWS IAM Service** with the following permissions:
    - `AmazonEC2FullAccess`
    - `AmazonS3FullAccess`

    *(You can also create a custom policy for more restricted access if needed.)*

2. **Generate an Access Key** for this IAM user and **save the following securely**:
    - Access Key ID
    - Secret Access Key

3. The S3 Bucket will be created dynamically during the execution of the [testing_pipeline.py](src/testing_pipeline.py). If the best found model during training achieves the required score in the [testing_config.json](config/testing_config.json), it will be uploaded to this bucket.\
⚠️ (*Make sure the bucket name you configure is globally unique to avoid conflicts.*)

4. Create an EC2 instance with the following specifications:
    - **AMI**: Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.7 (Ubuntu 22.04)
    - **Instance type**: `t3.medium` (`t2.micro` could result in a timeout during the execution of the Streamlit application [app.py](src/web_app/app.py))
    - **Key Pair**: Use the key pair you download to connect via SSH
    - **Security Group**:
        - Allow inbound rules for ports: `22`, `80`, `8501`, `8502` (all TCP)
    - **Storage**: 120 GiB (gp3)

5. Run the EC2 instance and interact with it via SSH or any other remote access method to set up the environment:
    - Create a working directory:
        ```bash
        mkdir mlops
        cd mlops
        ```
    - Clone your GitHub repo:
        ```bash
        git clone https://github.com/Lahdhirim/NLP_sentiment_classification_aws.git
        cd NLP_sentiment_classification_aws
        ```
    - Install dependencies:
        ```bash
        pip install -r requirements.txt
        ```
    - Configure the AWS credentials using AWS CLI:
        ```bash
        aws configure Press ENTER
        AWS Access Key ID: ************
        AWS Secret Access Key: ************
        Default region name: Press ENTER
        Default output format: Press ENTER
        ```
    - Add Streamlit to PATH (for command-line use):
        ```bash
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
        source ~/.bashrc
        ```

6. (Optional but recommended) Run the Streamlit application from the terminal to ensure that everything is working correctly:
    ```bash
    streamlit run src/web_app/app.py
    ```

7. Automatically launch Streamlit on instance reboot:
    - Create the startup script:
      ```bash
      nano /home/ubuntu/start_streamlit.sh
      ```

      Paste the following:
      ```bash
      #!/bin/bash
      cd /home/ubuntu/mlops/NLP_sentiment_classification_aws
      source /home/ubuntu/.bashrc
      /home/ubuntu/.local/bin/streamlit run src/web_app/app.py >> /home/ubuntu/streamlit.log 2>&1
      ```

    - Make the script executable:
        ```bash
        chmod +x /home/ubuntu/start_streamlit.sh 
        ```
    - Add the script to `crontab` for reboot:
        ```bash
        crontab -e
        ```
        Add the following line at the end of file:
        ```bash
        @reboot /home/ubuntu/start_streamlit.sh
        ```

Each time the instance is rebooted, Streamlit will automatically launch the web application at the address  `http://<public IPv4 address>:8501`. A log file named  `streamlit.log` will be created in the  `/home/ubuntu` directory. This file can be used to monitor the application’s status and debug any errors.\
The application will be publicly accessible to anyone with the instance’s public IP address. Access can be controlled via the EC2 Security Group:
- To allow access from any IP address, set the Source  `to 0.0.0.0/0` on TCP port  `8501`.\
    ⚠️ Use  `0.0.0.0/0` only if you're aware of the security implications. For more restricted access, specify your own IP or a limited range.

## (BONUS) Steps to reduce overfitting
- Freeze the backbone of the model during training. Note that keeping the last encoder layer (bert.encoder.layer.3) trainable allows for greater task-specific adaptation; otherwise, the classifier alone is too simple to capture complex patterns (Accuracy 66%). 
- Add dropout layer control in the configuration ([training_config.json](config/training_config.json)).
- Disable learning rate warmup, otherwise the learning rate can remind too low and results in a fast overfiting.











