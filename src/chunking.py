from __future__ import annotations
import math
import re

class FixedSizeChunker:
    """Đã được implement sẵn"""
    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
            
        # Tách câu dựa trên ". ", "! ", "? " hoặc ".\n"
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|(?<=\.)\n', text) if s.strip()]

        chunks = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            chunk_sentences = sentences[i : i + self.max_sentences_per_chunk]
            chunks.append(" ".join(chunk_sentences))
        return chunks


class RecursiveChunker:
    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if len(current_text) <= self.chunk_size:
            return [current_text]

        if not remaining_separators:
            # Fallback: cắt cứng theo độ dài nếu hết separator
            return [current_text[i:i+self.chunk_size] for i in range(0, len(current_text), self.chunk_size)]

        sep = remaining_separators[0]
        next_seps = remaining_separators[1:]

        splits = current_text.split(sep)
        chunks = []
        current_chunk = ""

        for piece in splits:
            piece_with_sep = piece if not current_chunk else sep + piece
            if len(current_chunk) + len(piece_with_sep) <= self.chunk_size:
                if not current_chunk:
                    current_chunk = piece
                else:
                    current_chunk += sep + piece
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = piece

        if current_chunk:
            chunks.append(current_chunk)

        # Đệ quy cho các phần vẫn còn vượt quá giới hạn
        final_chunks = []
        for c in chunks:
            if len(c) > self.chunk_size:
                final_chunks.extend(self._split(c, next_seps))
            else:
                final_chunks.append(c)

        return final_chunks


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))

def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    # Guard: Trả về 0.0 nếu có vector độ lớn = 0
    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


class ChunkingStrategyComparator:
    def compare(self, text: str, chunk_size: int = 200) -> dict:
        strategies = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size),
            "by_sentences": SentenceChunker(max_sentences_per_chunk=3),
            "recursive": RecursiveChunker(chunk_size=chunk_size)
        }

        results = {}
        for name, strategy in strategies.items():
            chunks = strategy.chunk(text)
            count = len(chunks)
            avg_length = sum(len(c) for c in chunks) / count if count > 0 else 0
            results[name] = {
                "count": count,
                "avg_length": avg_length,
                "chunks": chunks
            }
        return results