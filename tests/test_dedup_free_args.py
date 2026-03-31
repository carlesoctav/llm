import subprocess
from pathlib import Path

from jaxformers.dedup_free_args import dedup_command


def test_dedup_command_keeps_latest_override_values():
    command = (
        'HF_HOME="/mnt/carles/.cache" python src/jaxformers/train/ntp.py '
        "--config experiments/tunix-repro/base_config.py "
        "experiments/tunix-repro/config/lora_gemma3.py "
        "eval=None project_name=hf_dir c.dir=/mnt/carles/llm/ckptr/ "
        'exp_name=test_hf_1 model_name="{MODEL_DIR}.huggingface.gemma3.Gemma3ForCausalLM.from_pretrained" checkpoint=None '
        "weights_impl=free logger_name=noop weights_impl=stack "
        "forward_impl=scan_layer remat_layer=True logger_name=wandb "
        "project_name=tunix-again exp_name=new_eqx_module_new "
        "max_train_step=30 logger_name=noop max_train_step=10_000 "
        "weights_impl=free forward_impl=loop"
    )

    assert dedup_command(command) == (
        "HF_HOME=/mnt/carles/.cache python src/jaxformers/train/ntp.py "
        "--config experiments/tunix-repro/base_config.py "
        "experiments/tunix-repro/config/lora_gemma3.py "
        "eval=None c.dir=/mnt/carles/llm/ckptr/ "
        "'model_name={MODEL_DIR}.huggingface.gemma3.Gemma3ForCausalLM.from_pretrained' checkpoint=None remat_layer=True "
        "project_name=tunix-again exp_name=new_eqx_module_new "
        "logger_name=noop max_train_step=10_000 "
        "weights_impl=free forward_impl=loop"
    )


def test_dedup_script_reads_command_from_stdin():
    repo_root = Path(__file__).resolve().parents[1]
    command = "python train.py foo=1 bar=2 foo=3"

    completed = subprocess.run(
        [str(repo_root / "dedup.sh")],
        check=True,
        cwd=repo_root,
        input=command,
        text=True,
        capture_output=True,
    )

    assert completed.stdout == "python train.py bar=2 foo=3\n"
