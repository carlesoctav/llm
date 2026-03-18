import jax
import numpy as np
import safetensors
import torchax
from transformers import AutoModelForCausalLM


torchax.enable_globally()


def main():
    with torchax.disable_temporarily():
        hf_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")

    hf_model = hf_model.to("jax")
    jax_weight = None
    name, weight = next(hf_model.named_parameters())
    weight = weight.jax()

    with safetensors.safe_open(
        "/home/carlesoctav/weights/huggingface/Qwen/Qwen3-0.6B/model.safetensors",
        framework="np",
    ) as f:
        for key in f.keys():
            if key == name:
                jax_weight = jax.device_put(f.get_tensor(key))
                break

    np.testing.assert_allclose(jax_weight, weight)


if __name__ == "__main__":
    main()
