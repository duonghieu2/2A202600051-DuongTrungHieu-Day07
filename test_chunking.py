from src.chunking import ChunkingStrategyComparator

# Đọc nội dung 1 file tài liệu của nhóm để test
with open("data_v2/markdown/3.md", "r", encoding="utf-8") as f:
    text = f.read()

comparator = ChunkingStrategyComparator()
results = comparator.compare(text, chunk_size=500)

print("=== BASELINE ANALYSIS ===")
for strategy_name, stats in results.items():
    print(f"Chiến lược: {strategy_name}")
    print(f" - Số lượng chunk: {stats['count']}")
    print(f" - Chiều dài trung bình: {stats['avg_length']:.1f} ký tự\n")