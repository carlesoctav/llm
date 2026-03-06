from .llm import JaxformersLLM, LLM

__all__ = [
    "JaxformersLLM",
    "LLM",
    "build_gemma3_vllm_sync_payload",
    "pack_gemma3_for_vllm_sync",
]


def __getattr__(name: str):
    if name in {"build_gemma3_vllm_sync_payload", "pack_gemma3_for_vllm_sync"}:
        from .vllm_sync import build_gemma3_vllm_sync_payload, pack_gemma3_for_vllm_sync

        return {
            "build_gemma3_vllm_sync_payload": build_gemma3_vllm_sync_payload,
            "pack_gemma3_for_vllm_sync": pack_gemma3_for_vllm_sync,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
