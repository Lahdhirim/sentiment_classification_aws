from sklearn.metrics import accuracy_score
from src.utils.schema import MetricSchema

def compute_accuracy(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    acc = accuracy_score(labels, predictions)
    return {MetricSchema.ACCURACY: acc}