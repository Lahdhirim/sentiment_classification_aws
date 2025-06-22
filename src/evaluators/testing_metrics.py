import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import List, Dict, Any
from src.utils.schema import MetricSchema
from colorama import Fore, Style

class MetricsCalculator:
    def __init__(self, 
                 true_labels: List[str], 
                 pred_labels: List[str], 
                 output_csv_path: str
                 ) -> None:
        self.true_labels = true_labels
        self.pred_labels = pred_labels
        self.output_csv_path = output_csv_path

    def calculate_metrics(self) -> None:
        accuracy = accuracy_score(self.true_labels, self.pred_labels)
        precision = precision_score(self.true_labels, self.pred_labels, pos_label="positive")
        recall = recall_score(self.true_labels, self.pred_labels, pos_label="positive")
        f1 = f1_score(self.true_labels, self.pred_labels, pos_label="positive")

        metrics = {
            MetricSchema.ACCURACY: round(accuracy, 3),
            MetricSchema.PRECISION: round(precision, 3),
            MetricSchema.RECALL: round(recall, 3),
            MetricSchema.F1_SCORE: round(f1, 3),
        }

        df = pd.DataFrame([metrics])
        df.to_csv(self.output_csv_path, index=False)
        print(Fore.MAGENTA + f"CSV file with performance metrics saved at {self.output_csv_path}." + Style.RESET_ALL)