from typing import Callable
from .store import EmbeddingStore


class KnowledgeBaseAgent:
    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        # Bước 1: Trích xuất các kết quả văn bản liên quan
        retrieved_records = self.store.search(question, top_k=top_k)
        
        # Bước 2: Ghép các đoạn chunk lại với nhau
        context_blocks = [record["content"] for record in retrieved_records]
        context_str = "\n\n".join(context_blocks)
        
        # Bước 3: Tạo một Prompt hoàn chỉnh cho LLM
        prompt = (
            f"Sử dụng thông tin ngữ cảnh dưới đây để trả lời câu hỏi.\n"
            f"--- Context ---\n"
            f"{context_str}\n"
            f"---------------\n"
            f"Câu hỏi: {question}\n"
            f"Trả lời:"
        )
        
        # Bước 4: Trả về kết quả sau khi gọi hàm LLM callback
        return self.llm_fn(prompt)