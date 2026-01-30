from transformers import AutoModelForCausalLM
import torchax
torchax.enable_globally()


def main():
    with torchax.disable_temporarily():
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Base"

    model = model.to("jax")


if __name__ == "__main__":
    main()
