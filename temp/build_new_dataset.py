from datasets import load_dataset


dataset = load_dataset("carlesoctav/4b-generated-Dolci-Instruct-SFT-No-Tools", split = "train")


def fn(row):
    messages = []
    messages
    return {"messages": [{"role": "user", "content": row["prompt"]}, {"role": "assistant", "content": row["generated"]}]}


new_ds = dataset.map(fn, remove_columns = ["prompt", "generated"])
new_ds.push_to_hub("carlesoctav/4b-generated-Dolci-Instruct-SFT-No-Tools-messages")
