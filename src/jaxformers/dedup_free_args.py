import re
import shlex
import sys


ENV_ASSIGNMENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=.*")


def dedup_tokens(tokens: list[str]) -> list[str]:
    command_index = next(
        (i for i, token in enumerate(tokens) if not ENV_ASSIGNMENT_RE.fullmatch(token)),
        len(tokens),
    )
    prefix = tokens[:command_index]
    suffix = tokens[command_index:]

    if command_index == len(tokens):
        prefix = []
        suffix = tokens
    elif suffix:
        prefix.append(suffix[0])
        suffix = suffix[1:]

    seen = set()
    deduped_suffix = []
    for token in reversed(suffix):
        if "=" not in token or token.startswith("-"):
            deduped_suffix.append(token)
            continue

        key = token.split("=", 1)[0]
        if not key or key in seen:
            continue

        seen.add(key)
        deduped_suffix.append(token)

    deduped_suffix.reverse()
    return prefix + deduped_suffix


def dedup_command(command: str) -> str:
    stripped = command.strip()
    if not stripped:
        return ""

    return shlex.join(dedup_tokens(shlex.split(stripped)))


def main() -> int:
    if len(sys.argv) > 1:
        print(shlex.join(dedup_tokens(sys.argv[1:])))
        return 0

    command = sys.stdin.read()
    if not command.strip():
        return 0

    print(dedup_command(command))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
