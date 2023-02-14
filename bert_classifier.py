from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import TrainingArguments, Trainer
from sklearn.model_selection import train_test_split

import pandas as pd
from pandas import DataFrame
import numpy as np
import evaluate
from pathlib import Path


from prepare_data_set import prepare_data_set


MODEL_DIR = "model_data"
BASE_MODEL = "distilbert-base-cased"


class BERTMisogynyModel:

    def __init__(self, data_set: DataFrame, train_test_ratio: float):
        self.train_test_ratio: float = train_test_ratio
        self.prepared_data_set = self.marshall_dataset(data_set)
        self.trainer = self.get_trainer()

    def marshall_dataset(self, data_set: DataFrame):
        data_set = prepare_data_set(data_set)
        data_set = data_set.rename(columns={"Definition": "text", "is_misogyny": "label"})
        data_set['label'] = data_set['label'].astype(int)
        data_set.set_index('text', inplace=True)
        train_set, test_set = train_test_split(data_set, test_size=self.train_test_ratio)
        return DatasetDict({
            "train": Dataset.from_pandas(train_set), 
            "test": Dataset.from_pandas(test_set)
        })
    

    @staticmethod
    def get_tokenizer():
        return AutoTokenizer.from_pretrained(BASE_MODEL)
    
    def get_model(self):
        return AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)
    
    def get_training_arguments(self):
        return TrainingArguments(output_dir=MODEL_DIR, evaluation_strategy="epoch")
    
    def get_tokenized_data_set(self):
        return self.prepared_data_set.map(self.tokenize_function, batched=True)

    def get_trainer(self):
        return Trainer(
            model=self.get_model(),
            args=self.get_training_arguments(),
            train_dataset=self.get_tokenized_data_set()["train"],
            eval_dataset=self.get_tokenized_data_set()["test"],
            compute_metrics=self.compute_metrics
        )

    @staticmethod
    def metric():
        return evaluate.load("accuracy")
        
    @staticmethod
    def tokenize_function(examples):
        return BERTMisogynyModel.get_tokenizer()(examples["text"], padding="max_length", truncation=True)
    
    @staticmethod
    def compute_metrics(eval_pred):
        print(type(eval_pred))
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return BERTMisogynyModel.metric.compute(predictions=predictions, references=labels)


if __name__ == '__main__':

    data_file_path = Path['DATA_FILE_NAME'](__file__) / 'data' / 'ManualTag_Misogyny.csv'
    df = pd.read_csv('data/ManualTag_Misogyny.csv', encoding='latin-1')

    bmm = BERTMisogynyModel(df, 0.2)
    bmm.trainer.train()
