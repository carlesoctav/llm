from transformers import AutoModel


model = AutoModel.from_pretrained("google-bert/bert-base-uncased")
a = model.state_dict()
