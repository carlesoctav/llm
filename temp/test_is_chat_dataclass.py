from jaxformers.data.next_token_prediction import transforms
from jaxformers.data.huggingface import load


load_kwargs = [
    {
        "path": "allenai/Dolci-Instruct-SFT-No-Tools",
        # "name": "GovReport",
        "split": "train",
        # "streaming": True,
    }
]

dataset = load(load_kwargs)
