from transformers import BertTokenizer, BertForSequenceClassification
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("./bert-imdb-model")
model.to(device)
model.eval()

with open("test_review.txt", "r") as f:
    text = f.read()


inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    padding=True
).to(device)

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

pred_id = logits.argmax(dim=1).item()

label_map = {0: "Negative", 1: "Positive"}
pred = label_map[pred_id]

print("The review was", pred)
