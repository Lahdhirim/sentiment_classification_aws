# Sentiment Classifier Applicaton using **Hugging Face Transformers** and **AWS Services (S3, EC2)** (Under Construction)
✅​ Preprocessing Pipeline : load, clean and split data

✅​ Training Pipeline : Train **TinyBERT** model

✅​​ Testing Pipeline : Evaluate the trained model and push model to **S3** bucket using **Boto3** (if the model meets the specified accuracy threshold)

✅​​ Deploy the model on **Streamlit Share Server** (Public link: https://sentimentclassificationaws-nq67u8bs4qsjdwcqsa4lso.streamlit.app/)

​⏳ Deploy the model on **AWS EC2** instance and make prediction using **Streamlit** and **FastAPI**

## Steps to reduce overfitting
- Freeze the backbone of the model during training. Note that keeping the last encoder layer (bert.encoder.layer.3) trainable allows for greater task-specific adaptation; otherwise, the classifier alone is too simple to capture complex patterns (Accuracy 66%). 
- Add dropout layer control in the configuration.
- Disable learning rate warmup, otherwise the learning rate can remind too low and results in a fast overfiting.

