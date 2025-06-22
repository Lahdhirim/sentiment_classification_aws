# Sentiment Classifier Applicaton using **Hugging Face Transformers** and **AWS Services (S3, EC2)** (Under Construction)
✅​ Preprocessing Pipeline : load, clean and split data

✅​ Training Pipeline : Train **TinyBERT** model

✅​​ Testing Pipeline : Evaluate the trained model and push model to **S3** bucket using **Boto3** (if the model meets the specified accuracy threshold)

✅​​ Deploy the model on **Streamlit Share Server** (Public link: https://sentimentclassificationaws-nq67u8bs4qsjdwcqsa4lso.streamlit.app/)

![Demo](assets/demo_streamlit.gif)

​⏳ Deploy the model on **AWS EC2** instance and make prediction using **Streamlit** and **FastAPI**

## Steps to reduce overfitting
- Freeze the backbone of the model during training. Note that keeping the last encoder layer (bert.encoder.layer.3) trainable allows for greater task-specific adaptation; otherwise, the classifier alone is too simple to capture complex patterns (Accuracy 66%). 
- Add dropout layer control in the configuration.
- Disable learning rate warmup, otherwise the learning rate can remind too low and results in a fast overfiting.

## Examples
- Review: I recently watched this movie and was absolutely blown away by the performances. The story was heartwarming and the characters were incredibly well-developed. It kept me engaged from beginning to end, and I couldn’t stop smiling by the time it was over. Definitely one of the best films I’ve seen this year! ⇒ Prediction: positive (98.64%)
- Review: I contacted customer support for help with my account, but the experience was terrible. The representative was rude, unhelpful, and clearly not interested in resolving my issue. After waiting on hold for nearly an hour, I was told to check the FAQ instead. I left feeling frustrated and ignored ⇒ Prediction: negative (98.61%)

