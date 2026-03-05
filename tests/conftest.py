def pytest_configure():
    # Tokamax uses `absl.flags` internally and will try to parse `sys.argv` on
    # first access, which breaks under `pytest -q ...` due to unknown flags.
    # Pre-parse with a minimal argv so later accesses see `is_parsed=True`.
    try:
        from absl import flags  # type: ignore[import-not-found]

        if not flags.FLAGS.is_parsed():
            flags.FLAGS(["pytest"])
    except Exception:
        pass
