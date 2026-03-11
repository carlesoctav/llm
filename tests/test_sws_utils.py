from __future__ import annotations

import textwrap

from jaxformers.sws_utils import combine_and_write, load_config_builder, merge_config_builders, run


def write_config(path, body: str) -> None:
    path.write_text(textwrap.dedent(body))


def test_merge_config_builders_rebinds_lazies_to_merged_config(tmp_path):
    config_a = tmp_path / "model_config.py"
    write_config(
        config_a,
        """
        import sws

        def get_config():
            c = sws.Config()
            c.model.model_id = "model-a"
            c.tokenizer_name = lambda: c.model.model_id
            return c
        """,
    )

    config_b = tmp_path / "override_config.py"
    write_config(
        config_b,
        """
        import sws

        def get_config():
            c = sws.Config()
            c.model.model_id = "model-b"
            return c
        """,
    )

    builder = merge_config_builders([str(config_a), str(config_b)])
    final = builder.finalize([])

    assert final.model.model_id == "model-b"
    assert final.tokenizer_name == "model-b"


def test_run_applies_cli_overrides_after_all_config_files(tmp_path):
    config_a = tmp_path / "base_config.py"
    write_config(
        config_a,
        """
        import sws

        def get_config():
            c = sws.Config()
            c.value = 1
            c.value_from_lazy = lambda: c.value
            return c
        """,
    )

    config_b = tmp_path / "second_config.py"
    write_config(
        config_b,
        """
        import sws

        def get_config():
            c = sws.Config()
            c.value = 2
            return c
        """,
    )

    result = run(
        lambda config: (config.value, config.value_from_lazy),
        argv=["--config", str(config_a), str(config_b), "value=3"],
    )

    assert result == (3, 3)


def test_run_supports_repeated_config_flags(tmp_path):
    config_a = tmp_path / "config_a.py"
    write_config(
        config_a,
        """
        import sws

        def get_config():
            c = sws.Config()
            c.alpha = 1
            return c
        """,
    )

    config_b = tmp_path / "config_b.py"
    write_config(
        config_b,
        """
        import sws

        def get_config():
            c = sws.Config()
            c.beta = 2
            return c
        """,
    )

    result = run(
        lambda config: (config.alpha, config.beta),
        argv=["--config", str(config_a), "--config", str(config_b)],
    )

    assert result == (1, 2)


def test_combine_and_write_generates_dotted_assignments(tmp_path):
    config_a = tmp_path / "config_a.py"
    write_config(
        config_a,
        """
        import sws

        def get_config():
            c = sws.Config()
            c.model.model_id = "model-a"
            return c
        """,
    )

    config_b = tmp_path / "config_b.py"
    write_config(
        config_b,
        """
        import sws

        def get_config():
            c = sws.Config()
            c.model.model_id = "model-b"
            c.tokenizer_name = lambda: c.model.model_id
            return c
        """,
    )

    output_path = tmp_path / "combined.py"
    combine_and_write(output_path, config_a, config_b)

    generated = output_path.read_text()
    assert "config.model.model_id = " in generated
    assert "config.tokenizer_name = " in generated

    final = load_config_builder(str(output_path)).finalize(["model_id='model-c'"])
    assert final.model.model_id == "model-c"
    assert final.tokenizer_name == "model-c"


def test_combine_and_write_creates_parent_dir(tmp_path):
    config_a = tmp_path / "config_a.py"
    write_config(
        config_a,
        """
        import sws

        def get_config():
            c = sws.Config()
            c.value = 1
            return c
        """,
    )

    output_path = tmp_path / "nested" / "dir" / "combined.py"
    combine_and_write(output_path, config_a)

    assert output_path.exists()
