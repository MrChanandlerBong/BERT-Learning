from datasets import load_dataset
dataset = load_dataset('imdb')
import sys

import transformers

#Loading Tokeniser
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenise(batch):
    return tokenizer(
        batch['text'],
        padding = "max_length",
        truncation  = True,
        max_length = 256)

dataset = dataset.map(tokenise , batched = True)
dataset["train"].set_format(
    "torch",
    columns=["input_ids", "attention_mask", "label"]
)

dataset["test"].set_format(
    "torch",
    columns=["input_ids", "attention_mask", "label"]
)


from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels = 2)

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Using device:", device)

from sklearn.metrics import accuracy_score
import numpy as np

def compute_metrics(eval_pred):
    logits , labels = eval_pred
    predictions = np.argmax(logits , axis = 1)
    return {"accuracy":accuracy_score(labels,predictions)}

#Training

from transformers import TrainingArguments

training_arguments = TrainingArguments(
    output_dir = "./results",
    learning_rate = 0.00002,
    per_device_train_batch_size = 8,
    per_device_eval_batch_size = 8,
    num_train_epochs = 2)

from transformers import Trainer

trainer = Trainer(
    model = model,
    args = training_arguments,
    train_dataset = dataset["train"],
    eval_dataset = dataset["test"],
    compute_metrics  = compute_metrics)

trainer.train()
metrics = trainer.evaluate()
print(metrics)
f=open("Log.txt",'a')
for name,data in metrics.items():
    f.write(f"{name}:{data}\n")
f.write("\n\n")
f.close()

from sklearn.metrics import accuracy_score, classification_report
import numpy as np

preds = trainer.predict(dataset["test"])

y_true = preds.label_ids
y_pred = np.argmax(preds.predictions, axis=1)

acc = accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred)

with open("Report.txt", "w") as f:
    f.write(f"Accuracy: {acc}\n\n")
    f.write(report)


    

trainer.save_model("./bert-imdb-model")
tokenizer.save_pretrained("./bert-imdb-model")
    
