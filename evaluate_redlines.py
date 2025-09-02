import argparse
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from collections import defaultdict

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def best_match(snippet, clause_contents, clause_names, model):
    """Return index and score of best match using substring, else embedding."""
    for i, content in enumerate(clause_contents):
        if snippet and snippet.strip() and snippet.strip() in content:
            return i, 1.0, "substring"
    if not snippet or not snippet.strip():
        return -1, 0, "none"
    emb_snip = model.encode(snippet)
    emb_clauses = model.encode(clause_contents)
    scores = [cosine_sim(emb_snip, emb_clause) for emb_clause in emb_clauses]
    idx = int(np.argmax(scores))
    return idx, float(scores[idx]), "embedding"

def similarity(a, b, model):
    """Cosine similarity between two texts (embeddings)."""
    if not a or not b:
        return 0.0
    emb_a, emb_b = model.encode(a), model.encode(b)
    return cosine_sim(emb_a, emb_b)

def main():
    parser = argparse.ArgumentParser(description="NDA Clause-by-Clause Evaluation")
    parser.add_argument("--expected", required=True, help="expected_output.json")
    parser.add_argument("--actual", required=True, help="actual_output.json (redlined)")
    parser.add_argument("--clauses", required=True, help="bad_document_clauses.json")
    args = parser.parse_args()

    # Load files
    expected = load_json(args.expected)
    actual = load_json(args.actual)
    clauses = load_json(args.clauses)

    # Set up for matching
    clause_names = [c['clause_name'].strip() for c in clauses]
    clause_contents = [c['clause_content'] for c in clauses]
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Build dict: clause_name -> expected/actual details
    expected_map = {}
    for e in expected:
        idx, _, _ = best_match(e['text_snippet'], clause_contents, clause_names, model)
        cname = clause_names[idx] if idx >= 0 else "(unmatched)"
        expected_map[cname] = e

    actual_map = {}
    for a in actual:
        idx, _, _ = best_match(a['text_snippet'], clause_contents, clause_names, model)
        cname = clause_names[idx] if idx >= 0 else "(unmatched)"
        actual_map[cname] = a

    # For each canonical NDA clause, assemble evaluation
    lines = []
    playbook_refs_expected = set()
    playbook_refs_actual = set()
    tp, fp, fn = 0, 0, 0

    for clause in clauses:
        cname = clause['clause_name'].strip()
        e = expected_map.get(cname, None)
        a = actual_map.get(cname, None)
        # Playbook reference (may not be present)
        pb_e = e.get('playbook_clause_reference', "None") if e else "None"
        pb_a = a.get('playbook_clause_reference', "None") if a else "None"
        playbook_refs_expected.add(pb_e)
        playbook_refs_actual.add(pb_a)
        flagged_e = "YES" if e else "NO"
        flagged_a = "YES" if a else "NO"
        #text_sim = similarity(e['text_snippet'], a['text_snippet'], model) if (e and a) else 0.0
        #fix_sim = similarity(e.get('suggested_fix',''), a.get('suggested_fix',''), model) if (e and a) else 0.0

        if flagged_e == "NO" and flagged_a == "NO":
            text_sim = "N/A"
            fix_sim = "N/A"
        elif e and a:
            text_sim = f"{similarity(e['text_snippet'], a['text_snippet'], model):.2f}"
            fix_sim = f"{similarity(e.get('suggested_fix',''), a.get('suggested_fix',''), model):.2f}"
        else:
            text_sim = "0.00"
            fix_sim = "0.00"


        # For summary
        if flagged_e == "YES" and flagged_a == "YES":
            tp += 1
        elif flagged_e == "NO" and flagged_a == "YES":
            fp += 1
        elif flagged_e == "YES" and flagged_a == "NO":
            fn += 1
        lines.append({
            "clause": cname,
            "playbook_expected": pb_e,
            "playbook_actual": pb_a,
            "flagged_expected": flagged_e,
            "flagged_actual": flagged_a,
            "text_sim": text_sim,
            "fix_sim": fix_sim,
        })

    # Summary metrics
    precision = tp / (tp+fp) if (tp+fp) else 0
    recall = tp / (tp+fn) if (tp+fn) else 0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall) else 0

    # Output (title and metrics)
    print("="*70)
    print("NDA Clause Redline Evaluation")
    print("="*70)
    print(f"TP: {tp}   FP: {fp}   FN: {fn}   Precision: {precision:.2f}   Recall: {recall:.2f}   F1: {f1:.2f}")
    print("="*70)
    for row in lines:
        print(f"Clause: {row['clause']}")
        print(f"  - Expected:   Playbook: {row['playbook_expected']} | Flagged: {row['flagged_expected']}")
        print(f"  - Actual:     Playbook: {row['playbook_actual']} | Flagged: {row['flagged_actual']}")
        print(f"  - Text Snippet Similarity: {row['text_sim']}")
        print(f"  - Suggested Fix Similarity: {row['fix_sim']}")
        print("-"*60)

if __name__ == "__main__":
    main()
