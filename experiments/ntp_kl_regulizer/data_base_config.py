import sws
from transformers import AutoTokenizer


def get_config():
    config = sws.Config()
    config.data.loader.combine = "zip"
    config.data.loader.shard = False

    config.data.sft.source.load_kwargs = [
        {
            "path": "carlesoctav/4b-generated-Dolci-Instruct-SFT-No-Tools-messages",
            "split": "train",
        }
    ]
    config.data.sft.source.streaming = False
    config.data.kl.source.load_kwargs = [
        {
            "path": "carlesoctav/4b-generated-Dolci-Instruct-SFT-No-Tools-messages",
            "split": "train",
        }
    ]
    config.data.kl.source.streaming = False

    config.data.sft.transforms_name = "ntp"
    config.data.sft.transforms.column = "messages"
    config.data.sft.transforms.max_length = 2048
    config.data.sft.transforms.tokenizer = lambda: AutoTokenizer.from_pretrained(
        config.model.model_id
    )
    config.data.sft.transforms.data_type = "chat"
    config.data.sft.transforms.assistant_loss = True
    config.data.sft.transforms.chat_template_path = "./temp/think.jinja"
    config.data.sft.transforms.packing = False
    config.data.sft.transforms.packing_bins = 64

    config.data.kl.transforms_name = "kl"
    config.data.kl.transforms.column = "messages"
    config.data.kl.transforms.max_length = 2048
    config.data.kl.transforms.tokenizer = lambda: AutoTokenizer.from_pretrained(
        config.model.model_id
    )
    config.data.kl.transforms.data_type = "chat"
    config.data.kl.transforms.assistant_loss = True
    config.data.kl.transforms.chat_template_path = "./temp/think.jinja"
    config.data.kl.transforms.packing = False
    config.data.kl.transforms.packing_bins = 64

    config.data.sft.loader.batch_size = 32
    config.data.sft.loader.shuffle = False
    config.data.kl.loader.batch_size = 32
    config.data.kl.loader.shuffle = False
    config.data.loader.seed = 42

    return config
