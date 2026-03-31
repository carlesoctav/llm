from jaxformers.modeling_utils import resolve_model_dir
from jaxformers.models import make_model
from jaxformers.models import model_accepts_kwarg


def helper_factory(*args, **kwargs):
    return args, kwargs


class DummyModel:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls, args, kwargs


def test_make_model_accepts_callable_target():
    out_args, out_kwargs = make_model(helper_factory, "repo", revision="main")

    assert out_args == ("repo",)
    assert out_kwargs == {"revision": "main"}


def test_make_model_resolves_dotted_callable_target():
    out_args, out_kwargs = make_model(
        "test_make_model.helper_factory",
        "repo",
        revision="main",
    )

    assert out_args == ("repo",)
    assert out_kwargs == {"revision": "main"}


def test_make_model_resolves_dotted_classmethod_target():
    out_cls, out_args, out_kwargs = make_model(
        "test_make_model.DummyModel.from_pretrained",
        "repo",
        revision="main",
    )

    assert out_cls is DummyModel
    assert out_args == ("repo",)
    assert out_kwargs == {"revision": "main"}


def test_model_accepts_kwarg_detects_named_kwarg():
    assert model_accepts_kwarg(helper_factory, "revision") is True
    assert model_accepts_kwarg(helper_factory, "rngs") is True


def test_resolve_model_dir_prefers_explicit_arg(monkeypatch):
    monkeypatch.setenv("MODEL_DIR", "test_make_model")

    assert resolve_model_dir("test_explicit") == "test_explicit"


def test_resolve_model_dir_uses_env(monkeypatch):
    monkeypatch.setenv("MODEL_DIR", "test_make_model")

    assert resolve_model_dir() == "test_make_model"


def test_make_model_resolves_model_dir_placeholder(monkeypatch):
    monkeypatch.setenv("MODEL_DIR", "test_make_model")

    out_cls, out_args, out_kwargs = make_model(
        "{MODEL_DIR}.DummyModel.from_pretrained",
        "repo",
        revision="main",
    )

    assert out_cls is DummyModel
    assert out_args == ("repo",)
    assert out_kwargs == {"revision": "main"}
