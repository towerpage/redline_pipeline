import os
import json
from datetime import datetime
from dotenv import load_dotenv
import openai
import time

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
assert openai.api_key, "OPENAI_API_KEY not found in .env"

# Load files
with open("playbook.json", "r", encoding="utf-8") as f:
    playbook = json.load(f)
with open("bad_document_clauses.json", "r", encoding="utf-8") as f:
    nda_clauses = json.load(f)

playbook_names = [pb.get("clause", "[NO NAME]") for pb in playbook]

match_matrix = {}  # playbook clause -> {nda_idx: score}

for pb_idx, pb in enumerate(playbook):
    pb_name = pb.get("clause", "[NO NAME]")
    pb_heading = pb_name
    pb_def = pb.get("clause_definition", "")
    match_matrix[pb_name] = {}
    print(f"\nScoring Playbook Clause: {pb_name}")

    for nda_idx, nda in enumerate(nda_clauses):
        nda_heading = nda.get("clause_name", "[NO NAME]")
        nda_content = nda.get("clause_content", "[NO CONTENT]")

        # Build scoring prompt
        prompt = f"""
You are a legal contract clause mapping expert.

For each pair below, score the match on a scale from 0 (not a match at all) to 10 (perfect match). Consider both the playbook clause name/definition and the NDA clause heading/content. A match means the NDA clause serves the *same legal purpose* as the playbook clause.

Playbook Clause Name: {pb_heading}
Playbook Clause Definition: {pb_def}

NDA Clause Name: {nda_heading}
NDA Clause Content: {nda_content}

Score (0–10), and nothing else:
"""
        response = openai.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": "You are a legal contract clause mapping expert."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4,
            temperature=0,
        )
        answer = response.choices[0].message.content.strip()
        try:
            score = int(answer.split()[0])
        except Exception:
            try:
                score = float(answer.split()[0])
            except Exception:
                print(f"WARNING: Could not parse score from LLM output: '{answer}'. Defaulting to 0.")
                score = 0
        match_matrix[pb_name][nda_idx] = score
        print(f"  NDA Clause #{nda_idx+1} [{nda_heading}]: {score}")

        time.sleep(0.3)  # avoid rate limit

# Print heatmap
print("\n=== MATCH SCORE MATRIX ===")
for pb_name in match_matrix:
    row = [str(match_matrix[pb_name][idx]) for idx in range(len(nda_clauses))]
    print(f"{pb_name:35s}: " + "  ".join(row))

# Save to file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
outfile = f"match_matrix_{timestamp}.json"
with open(outfile, "w", encoding="utf-8") as f:
    json.dump(match_matrix, f, indent=2)
print(f"\nWrote match matrix to: {outfile}")


# ======== DETERMINISTIC ASSIGNMENT BASED ONE HEATMAP =========

import numpy as np

nda_count = len(nda_clauses)
pb_names_list = list(match_matrix.keys())
score_mat = np.array([[match_matrix[pb][nda_idx] for nda_idx in range(nda_count)] for pb in pb_names_list])

assigned_nda = set()
final_mapping = {}

for pb_idx, pb_name in enumerate(pb_names_list):
    # Mask out already assigned NDA clauses
    scores = list(score_mat[pb_idx])
    for i in assigned_nda:
        scores[i] = -1e9  # Impossible score, so can't be picked
    best_score = max(scores)
    if best_score >= 6:  # Set threshold for "real" match (tune as needed)
        nda_idx = scores.index(best_score)
        final_mapping[pb_name] = nda_idx
        assigned_nda.add(nda_idx)
        print(f"{pb_name}: NDA Clause #{nda_idx+1} — Score: {best_score}")
    else:
        final_mapping[pb_name] = None
        print(f"{pb_name}: None (no match ≥ threshold)")

# Save mapping to file
mapfile = f"playbook_to_nda_mapping_{timestamp}.json"
with open(mapfile, "w", encoding="utf-8") as f:
    json.dump(final_mapping, f, indent=2)
print(f"\nWrote final mapping to: {mapfile}")


# ==== SECOND PASS: Focused LLM assignment for unmapped playbook clauses ====

# Find unmapped playbook clause names
unmapped_pb_names = [pb for pb, nda_idx in final_mapping.items() if nda_idx is None]

if unmapped_pb_names:
    print("\n=== Second Pass: LLM focused matching for unmapped playbook clauses ===")
else:
    print("\n=== All playbook clauses mapped in first pass. Skipping second pass. ===")

# Build set of already-mapped NDA clause indices
already_mapped_nda = set(idx for idx in final_mapping.values() if idx is not None)

for pb_name in unmapped_pb_names:
    pb_record = next((pb for pb in playbook if pb.get("clause") == pb_name), None)
    if not pb_record:
        print(f"WARNING: Playbook record for unmapped clause '{pb_name}' not found.")
        continue
    clause_definition = pb_record.get("clause_definition", "")
    red_flag = pb_record.get("red_flag", "")
    review_instruction = pb_record.get("review_instruction", "")
    # Gather unmapped NDA clauses
    unmapped_nda_indices = [i for i in range(len(nda_clauses)) if i not in already_mapped_nda]
    unmapped_nda_text = []
    for idx in unmapped_nda_indices:
        heading = nda_clauses[idx].get("clause_name", "[NO NAME]")
        content = nda_clauses[idx].get("clause_content", "[NO CONTENT]")
        unmapped_nda_text.append(f"{idx+1}. Heading: {heading}\n   Content: {content}")
    unmapped_nda_block = "\n\n".join(unmapped_nda_text)
    # Build prompt for this playbook clause
    prompt = f"""
You are a legal contract clause analyst.

Here is a playbook clause type, with definition and red flag(s):

Clause Name: {pb_name}
Clause Definition: {clause_definition}
Red Flag: {red_flag}
Review Instruction: {review_instruction}

Here are all remaining unmapped NDA clauses (with number, heading, content):

{unmapped_nda_block}

Instructions:
- For the playbook clause above, if any NDA clause below matches the legal concept (even if heading is different), output "NDA Clause #{{n}}" (with n = number as shown above).
- If none clearly matches, output "None".
- Do not output anything else.
"""
    print(f"\nSecond pass LLM prompt for unmapped playbook clause: {pb_name}")
    response = openai.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": "You are a legal contract clause analyst."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=16,
        temperature=0,
    )
    answer = response.choices[0].message.content.strip()
    print(f"Second pass LLM answer for '{pb_name}': {answer}")

    if answer.lower().startswith("nda clause #"):
        try:
            nda_num = int(answer.replace("NDA Clause #", "").strip())
            nda_idx = nda_num - 1
            if nda_idx in already_mapped_nda:
                print(f"WARNING: NDA clause #{nda_num} already mapped! Skipping.")
                continue
            final_mapping[pb_name] = nda_idx
            already_mapped_nda.add(nda_idx)
            print(f"Second pass: Assigned '{pb_name}' to NDA clause #{nda_num}")
        except Exception as e:
            print(f"WARNING: Could not parse NDA clause number from LLM answer: '{answer}'. Skipping.")
    elif answer.lower() == "none":
        print(f"Second pass: '{pb_name}' remains unmapped.")
    else:
        print(f"WARNING: Unexpected second pass LLM answer for '{pb_name}': '{answer}'")

# Save final mapping after second pass
mapfile2 = f"playbook_to_nda_mapping_second_pass_{timestamp}.json"
with open(mapfile2, "w", encoding="utf-8") as f:
    json.dump(final_mapping, f, indent=2)
print(f"\nWrote mapping after second pass to: {mapfile2}")
