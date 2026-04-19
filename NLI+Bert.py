import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer)
from sklearn.metrics import accuracy_score

dataset = load_dataset("json",data_files = "train.json")


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


print(tokenizer.vocab_size)
def tokenize(batch):
    return tokeniser(
        batch["headlines"],
        padding = "max_length",
        truncation = True,
        max_length = 128)

#formatting dataset and creating tensors

dataset = dataset.map(tokenize , batched = True)

dataset["train"].set_format("torch",columns = ["input_ids" , "attention_masks" ,"label"])

dataset["validation"].set_format("torch",columns = ["input_ids" , "attention_masks" ,"label"])


#model declaration

model = BertForSequenceClassification.from_pretrained("bert-base-uncased",num_labels=2)

#Device Setting

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Using device:", device)

#Evaluating

def compute_metrics(eval_pred):
    logits , labels = eval_pred
    preds = np.argmax(logits , axis=1)
    return {"accuracy" : accuracy_score(labels,preds)}

#Testing and Training
training_args=TrainingArguments(
    output_dir = "./results",
    learning_rate = 1e-5,
    per_device_train_batch_size = 16,
    per_device_eval_batch_size = 16,
    num_train_epochs = 4)


trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = dataset["train"],
    eval_dataset = dataset["test"],
    compute_metrics = compute_metrics)

trainer.train()

trainer.evaluate()

predictions = trainer.predict(dataset["test"])

y_pred = np.argmax(predictions.predictions, axis=1)
y_true = predictions.label_ids

df_errors = pd.Dataframe({
    "headline":dataset["test"]["headline"],
    "Original Label :":y_true,
    "Model Predicted :":y_pred,})

errors = df_errors[df_errors["Original Label :"] != df_errors["Model Predicted :"]]
print("No. Of Errors :",len(errors))

false_pos = errors[(errors["Original Label :"]==0) and (errors["Model Predicted :"]==1)]
print("No. Of False Positives :",len(false_pos))

false_neg = errors[(errors["Original Label :"]==1) and (errors["Model Predicted :"]==0)]
print("No. Of False Negatives :",len(false_neg))

f = open("False_Positive.txt","w")
for k in range(len(false_pos)):
    f.write(false_pos[k]["headline"])
f.close()

f = open("False_Negative.txt","w")
for k in range(len(false_neg)):
    f.write(false_neg[k]["headline"])
f.close()

