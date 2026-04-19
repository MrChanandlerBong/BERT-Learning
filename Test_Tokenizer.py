from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
print("Vocab size:", tokenizer.vocab_size)
