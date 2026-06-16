import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score
import random

seed = 67
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


dataset = load_dataset("json", data_files="train.json")
dataset = dataset["train"].train_test_split(test_size=0.2)
dataset = dataset.rename_column("is_sarcastic", "label")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
print(tokenizer.vocab_size)

def tokenize(batch):
    return tokenizer(
        batch["headline"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

dataset = dataset.map(tokenize, batched=True)
test_headlines = list(dataset["test"]["headline"])

dataset["train"].set_format("torch", columns=["input_ids", "attention_mask", "label"])
dataset["test"].set_format("torch", columns=["input_ids", "attention_mask", "label"])

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Using device:", device)

def compute_metrics(eval_pred):
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    preds = np.argmax(logits, axis=1)
    return {"accuracy": accuracy_score(labels, preds)}

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=1e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate()
trainer.save_model("./sarcasm-bert")
tokenizer.save_pretrained("./sarcasm-bert")

predictions = trainer.predict(dataset["test"])
y_pred = np.argmax(predictions.predictions, axis=1)
y_true = predictions.label_ids

df_errors = pd.DataFrame({
    "headline": test_headlines,
    "Original Label": y_true,
    "Model Predicted": y_pred
})

errors = df_errors[df_errors["Original Label"] != df_errors["Model Predicted"]]
print("No. Of Errors :", len(errors))

false_pos = errors[
    (errors["Original Label"] == 0) &
    (errors["Model Predicted"] == 1)
]
print("No. Of False Positives :", len(false_pos))

false_neg = errors[
    (errors["Original Label"] == 1) &
    (errors["Model Predicted"] == 0)
]
print("No. Of False Negatives :", len(false_neg))

with open("False_Positive.txt", "w") as f:
    for text in false_pos["headline"]:
        f.write(text + "\n")

with open("False_Negative.txt", "w") as f:
    for text in false_neg["headline"]:
        f.write(text + "\n")

from sklearn.metrics import classification_report

print(classification_report(y_true, y_pred))

        
