from rag.ingest import load_baseline_db, load_docling_db

baseline = load_baseline_db()
docl = load_docling_db()

query = "what is the dose of amoxicillin for children"

baseline_ans = baseline.similarity_search(query, k=3)
docl_ans = docl.similarity_search(query, k=3)


def print_results(label, results):
    print(f"\n{'='*20} {label} {'='*20}")
    for i, doc in enumerate(results, start=1):
        source = doc.metadata.get("source", "unknown")
        print(f"\n--- Match {i} (source: {source}) ---")
        print(doc.page_content)


print_results("BASELINE", baseline_ans)
print_results("DOCLING", docl_ans) 