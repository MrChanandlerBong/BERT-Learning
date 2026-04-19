import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification

MODEL_PATH = "./sarcasm-bert"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()
def predict_sarcasm(text, threshold=0.5):
    """
    Predict sarcasm for a given text.

    Returns:
        label (int): 1 = sarcastic, 0 = not sarcastic
        confidence (float): probability of sarcasm
    """

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)

    sarcasm_prob = probs[0][1].item()
    label = int(sarcasm_prob >= threshold)

    return label, sarcasm_prob

f = open('test_text.txt','r')
predict_sarcasm(f.read())
