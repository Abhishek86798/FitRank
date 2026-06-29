import json
import numpy as np
from pathlib import Path
from src.precompute import _load_model, embed_texts, BGE_QUERY_PREFIX
from src.retriever import retrieve_top_k

# JD Variations representing real-world recruiter paraphrasing
JD_VARIANTS = {
    "baseline": """Senior AI Engineer. Redrob AI. Build ranking, retrieval, matching systems.
Requirements: Production experience with vector databases (Pinecone, Qdrant), embeddings (sentence-transformers), Python.
Evaluation framework experience: NDCG, MRR. Nice to have: LLMs, LambdaMART.
""",
    "var_1_synonyms": """Lead Machine Learning Engineer. Redrob Artificial Intelligence. Develop search, recsys, and ranking pipelines.
Requirements: Deployed vector search engines (Milvus, FAISS), dense embeddings (BGE), Python.
Metrics expertise: Offline evaluation, MAP. Nice to have: Large Language Models, XGBoost.
""",
    "var_2_restructured": """Redrob AI is looking for a Senior AI Engineer to own our matching and ranking infrastructure.
Must have strong Python and production experience with embeddings (e5, sentence-transformers) and vector databases.
You should know how to build evaluation frameworks (A/B testing, NDCG).
Bonus: Learning-to-rank, LLM fine-tuning.
""",
    "var_3_terse": """Senior AI Engineer.
- Ranking, retrieval, search.
- Vector DBs (Qdrant, Pinecone, FAISS).
- Embeddings (sentence-transformers).
- Python.
- Evaluation metrics (NDCG, MRR).
Bonus: LLMs, PEFT, LambdaMART.
"""
}

def calculate_jaccard(list1, list2):
    """Calculate intersection over union for two lists."""
    s1 = set(list1)
    s2 = set(list2)
    if len(s1.union(s2)) == 0:
        return 0.0
    return len(s1.intersection(s2)) / len(s1.union(s2))

def run_audit(artifacts_dir="artifacts", prefix="sample_"):
    emb_path = Path(artifacts_dir) / f"{prefix}embeddings.npy"
    ids_path = Path(artifacts_dir) / f"{prefix}candidate_ids.npy"
    
    if not emb_path.exists() or not ids_path.exists():
        print(f"Embeddings not found at {emb_path}. Please run precompute.py first.")
        return

    print("Loading precomputed embeddings...")
    embeddings = np.lib.format.open_memmap(emb_path, mode="r", dtype=np.float16)
    candidate_ids = np.load(ids_path, allow_pickle=True)
    
    model = _load_model()
    
    top_k_lists = {}
    
    for name, text in JD_VARIANTS.items():
        print(f"Embedding JD variant: {name}...")
        vec = embed_texts(model, [BGE_QUERY_PREFIX + text], show_progress=False)
        top_ids, _ = retrieve_top_k(embeddings, vec, candidate_ids, k=20)
        top_k_lists[name] = top_ids
        
    baseline_list = top_k_lists["baseline"]
    
    scores = []
    print("\nRobustness Audit Results (Jaccard Overlap with Baseline):")
    for name, top_ids in top_k_lists.items():
        if name == "baseline": 
            continue
        score = calculate_jaccard(baseline_list, top_ids)
        scores.append(score)
        print(f" - {name}: {score:.2%}")
        
    final_score = float(np.mean(scores)) if scores else 0.0
    print(f"\nFinal System Stability Score: {final_score:.2%}")
    
    report = {
        "stability_score": final_score,
        "variants": {name: calculate_jaccard(baseline_list, top_ids) for name, top_ids in top_k_lists.items() if name != "baseline"}
    }
    
    out_path = Path("eval/robustness_report.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Report saved to {out_path}")

if __name__ == "__main__":
    run_audit()
