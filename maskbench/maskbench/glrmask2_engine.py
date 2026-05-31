from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import numpy as np

from .engine import Engine


def _bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1))
    bs += list(range(ord("\u00a1"), ord("\u00ac") + 1))
    bs += list(range(ord("\u00ae"), ord("\u00ff") + 1))
    cs = bs[:]
    n = 0
    for byte in range(2**8):
        if byte not in bs:
            bs.append(byte)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


_UNICODE_TO_BYTE = {value: key for key, value in _bytes_to_unicode().items()}


def _token_str_to_bytes(token_str: str) -> bytes:
    return bytes(_UNICODE_TO_BYTE[ch] for ch in token_str)


def _find_glrmask_checkout() -> Path | None:
    repo_root = Path(__file__).resolve().parents[2]
    workspace_root = repo_root.parent
    env_checkout = os.environ.get("GLRMASK2_CHECKOUT")
    candidates = [
        Path(env_checkout).expanduser() if env_checkout else None,
        workspace_root / "glrmask2-jsonschemabench-integration",
        workspace_root / "glrmask2",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    return None


def _glrmask_checkout_version() -> str | None:
    checkout = _find_glrmask_checkout()
    if checkout is None:
        return None
    git = ["git", "-c", f"safe.directory={checkout}", "-C", str(checkout)]
    try:
        rev = subprocess.check_output(
            [*git, "rev-parse", "--short=12", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        branch = subprocess.check_output(
            [*git, "branch", "--show-current"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None
    return f"{branch}@{rev}" if branch else rev


def _import_glrmask():
    try:
        import _glrmask as glrmask

        return glrmask
    except ImportError as exc:
        checkout = _find_glrmask_checkout()
        hint = ""
        if checkout is not None:
            hint = (
                f" Build/install it in your active environment with: "
                f"maturin develop --manifest-path {checkout / 'python' / 'Cargo.toml'}"
            )
        raise RuntimeError(f"Could not import _glrmask.{hint}") from exc


class GlrMask2Engine(Engine):
    def __init__(
        self, *, multithreaded: bool = False, compile_threads: int | None = None
    ):
        super().__init__()
        self.multithreaded = multithreaded
        self.compile_threads = compile_threads
        self.glrmask = None
        self.vocab = None
        self.constraint = None
        self.state = None
        self.mask_data = None

    def get_id(self):
        return "glrmask2-mt" if self.multithreaded else "glrmask2"

    def get_name(self):
        if not self.multithreaded:
            return "glrmask2"
        if self.compile_threads is None:
            return "glrmask2-mt"
        return f"glrmask2-mt-{self.compile_threads}"

    def get_module(self):
        return "_glrmask"

    def get_version(self):
        checkout_version = _glrmask_checkout_version()
        if checkout_version is not None:
            return checkout_version
        return getattr(self.glrmask, "__version__", "dev")

    def init(self):
        if self.multithreaded:
            if self.compile_threads is not None:
                os.environ["GLRMASK_COMPILE_THREADS"] = str(self.compile_threads)
                os.environ["RAYON_NUM_THREADS"] = str(self.compile_threads)
        else:
            os.environ["GLRMASK_COMPILE_THREADS"] = "1"
            os.environ["RAYON_NUM_THREADS"] = "1"

        self.glrmask = _import_glrmask()

        id_to_token_bytes: dict[int, bytes] = {}
        for piece, token_id in self.tokenizer.get_vocab().items():
            try:
                id_to_token_bytes[int(token_id)] = _token_str_to_bytes(piece)
            except KeyError:
                id_to_token_bytes[int(token_id)] = piece.encode("utf-8")

        for special_id in getattr(self.tokenizer, "all_special_ids", []):
            if special_id in id_to_token_bytes:
                continue
            piece = self.tokenizer.convert_ids_to_tokens(special_id)
            if piece is None:
                id_to_token_bytes[special_id] = b""
                continue
            try:
                id_to_token_bytes[special_id] = _token_str_to_bytes(piece)
            except KeyError:
                id_to_token_bytes[special_id] = piece.encode("utf-8")

        self.vocab = self.glrmask.Vocab.from_id_to_bytes(id_to_token_bytes)
        prepare = getattr(self.glrmask, "prepare_vocab_for_compile", None)
        if prepare is not None:
            prepare(self.vocab)

    def compile_grammar(self, schema: dict):
        if self.vocab is None:
            raise RuntimeError("init() must run before compile_grammar()")
        schema_json = json.dumps(schema)
        self.constraint = self.glrmask.Constraint.from_json_schema(
            schema_json, self.vocab
        )
        self.state = self.constraint.start()
        self.mask_data = np.zeros(self.constraint.mask_len(), dtype=np.int32)

    def reset(self):
        if self.constraint is None:
            raise RuntimeError("No grammar compiled yet")
        self.state = self.constraint.start()

    def compute_mask(self):
        if self.state is None or self.mask_data is None:
            raise RuntimeError("No state initialized")
        self.mask_data.fill(0)
        self.state.fill_mask(self.mask_data)

    def commit_token(self, token: int) -> bool:
        if self.state is None or self.mask_data is None:
            raise RuntimeError("compute_mask must be called before commit_token")

        word = token // 32
        ok = (
            word < len(self.mask_data)
            and (int(self.mask_data[word]) & (1 << (token % 32))) != 0
        )
        if ok:
            self.state.commit_token(token)
        return ok
