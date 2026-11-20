import hashlib
import importlib
import math
from collections.abc import Iterable
from typing import Protocol, cast


class _SentenceEncoder(Protocol):
    def encode(
        self, text: str, convert_to_numpy: bool, normalize_embeddings: bool
    ) -> Iterable[float]: ...


class _SentenceTransformersModule(Protocol):
    SentenceTransformer: "_SentenceTransformerFactory"


class _SentenceTransformerFactory(Protocol):
    def __call__(self, model_name: str, *, device: str) -> _SentenceEncoder: ...


class Embedder(Protocol):
    def embed(self, text: str) -> tuple[float, ...]: ...


class HashEmbedder:
    def __init__(self, dimensions: int = 8) -> None:
        self._dimensions = dimensions

    def embed(self, text: str) -> tuple[float, ...]:
        values = [0.0] * self._dimensions
        words = text.lower().split()
        for index, word in enumerate(words):
            digest = hashlib.sha256(word.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self._dimensions
            sign = 1.0 if digest[4] % 2 else -1.0
            values[bucket] += sign / math.sqrt(index + 1)
        norm = math.sqrt(sum(value * value for value in values))
        return tuple(value / norm for value in values) if norm else tuple(values)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            module = importlib.import_module("sentence_transformers")
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for the live CPU embedder"
            ) from exc
        library = cast(_SentenceTransformersModule, module)
        self._model = library.SentenceTransformer(model_name, device="cpu")

    def embed(self, text: str) -> tuple[float, ...]:
        return tuple(
            self._model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        )
