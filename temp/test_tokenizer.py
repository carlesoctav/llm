from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
a = tokenizer("hallo dunia", return_tensors = "np")
print(a["attention_mask"])
