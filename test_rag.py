from vqa.pubmed_retriever.context_builder import build_context

print("--- RAG PubMed Context Test ---")
context = build_context("Is there cardiomegaly?", {"Cardiomegaly": 0.85, "Effusion": 0.1})
print("\n[Generated Context]")
print(context)
