import sws
from transformers import AutoTokenizer


def get_config():
    config = sws.Config()

    config.data.source_name = "huggingface"
    config.data.source.load_kwargs = [
        {
            "path": "carlesoctav/4b-generated-Dolci-Instruct-SFT-No-Tools-messages",
            "split": "train",
            "streaming": False,
        }
    ]

    config.data.transforms_name = "ntp_kl"

    config.data.transforms.column = "messages_sft"
    config.data.transforms.kl_column = "messages_kl"

    config.data.transforms.max_length = 2048
    config.data.transforms.tokenizer = lambda: AutoTokenizer.from_pretrained(
        config.model.model_id
    )
    config.data.transforms.data_type = "chat"
    config.data.transforms.assistant_loss = True

    config.data.transforms.chat_template_path = "./temp/think.jinja"

    config.data.transforms.packing = False
    config.data.transforms.packing_bins = 64

    config.train_loader_name = "simple"
    config.train_loader.global_batch_size = 32
    config.train_loader.seed = 42
    config.train_loader.shuffle = False
    config.train_loader.drop_remainder = True

    return config
