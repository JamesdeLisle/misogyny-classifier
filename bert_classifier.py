from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import TrainingArguments, Trainer
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import evaluate
from pathlib import Path


from prepare_data_set import prepare_data_set

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

model = 'distilbert-base-cased'

dataset = load_dataset('yelp_review_full')
data_file_path = Path['DATA_FILE_NAME'](__file__) / 'data' / 'ManualTag_Misogyny.csv'
df = pd.read_csv('data/ManualTag_Misogyny.csv', encoding='latin-1')
data_set = prepare_data_set(df)
data_set = data_set.rename(columns={"Definition": "text", "is_misogyny": "label"})
data_set['label'] = data_set['label'].astype(int)
data_set.set_index('text', inplace=True)
train_set, test_set = train_test_split(data_set, test_size=0.2)
data_set_dict = DatasetDict({
    "train": Dataset.from_pandas(train_set), 
    "test": Dataset.from_pandas(test_set)
})

tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModelForSequenceClassification.from_pretrained(model, num_labels=2)
training_args = TrainingArguments(output_dir="test_trainer")
metric = evaluate.load("accuracy")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = data_set_dict.map(tokenize_function, batched=True)

training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
)

trainer.train()