from __future__ import annotations

from jaxformers.sws_utils import load_config_builder
from jaxformers.train.make_sweep import get_config, main


def test_make_sweep_copies_base_config_and_generated_runs_use_copy(tmp_path):
    base_config_path = tmp_path / "base_config.py"
    base_config_path.write_text(
        "\n".join(
            [
                "import sws",
                "",
                "def get_config():",
                "    c = sws.Config()",
                "    c.value = 7",
                "    return c",
                "",
            ]
        )
    )

    builder = get_config()
    builder.base_config = str(base_config_path)
    builder.dir = str(tmp_path / "sweeps")
    builder.name = "demo"
    config = builder.finalize([])

    main(config)

    out_dir = tmp_path / "sweeps" / "demo"
    copied_base_config_path = out_dir / base_config_path.name
    generated_config_path = out_dir / "0.py"

    assert copied_base_config_path.exists()
    assert copied_base_config_path.read_text() == base_config_path.read_text()

    base_config_path.unlink()

    final = load_config_builder(str(generated_config_path)).finalize([])
    assert final.value == 7
