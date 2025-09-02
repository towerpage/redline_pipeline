import os
import json
import re
from datetime import datetime
from dotenv import load_dotenv
import openai

# --- Load OpenAI API key from .env ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
assert OPENAI_API_KEY, "OPENAI_API_KEY not found in .env"


openai.api_key = OPENAI_API_KEY

#========= Load Inputs ============

# --- Load bad_document_clauses.json ---
with open("bad_document_clauses.json", "r", encoding="utf-8") as f:
    bad_clauses = json.load(f)

# --- Load playbook.json ---
with open("playbook.json", "r", encoding="utf-8") as f:
    playbook = json.load(f)

print(f"Loaded {len(bad_clauses)} document clauses")
print(f"Loaded {len(playbook)} playbook clause types")

# --- Prepare NDA clause list for prompt ---
nda_clause_blocks = []
for idx, clause in enumerate(bad_clauses):
    clause_heading = clause.get('clause_name', '[NO NAME]')
    clause_content = clause.get('clause_content', '[NO CONTENT]')
    nda_clause_blocks.append(
        f"{idx+1}. Heading: {clause_heading}\n   Content: {clause_content}"
    )
    print(clause_heading)
nda_clause_list_text = "\n\n".join(nda_clause_blocks)

# --- Prepare playbook clause list for prompt ---
playbook_refs = []
for clause in playbook:
    clause_name = clause.get("clause", "[NO NAME]")
    clause_def = clause.get("clause_definition", "[NO DEFINITION]")
    playbook_refs.append(f"- {clause_name}: {clause_def}")
playbook_reference_text = "\n".join(playbook_refs)



# --- Load playbook-to-NDA mapping ---
#PLAYBOOK_TO_NDA_MAPPING_FILEPATH = "playbook_to_nda_mapping_20250830_192308.json"
PLAYBOOK_TO_NDA_MAPPING_FILEPATH = "playbook_to_nda_mapping_second_pass_20250831_084755.json"
with open(PLAYBOOK_TO_NDA_MAPPING_FILEPATH, "r", encoding="utf-8") as f:
    playbook_to_nda_idx = json.load(f)
print("Loaded mapping (playbook clause name → NDA clause idx):")
print(playbook_to_nda_idx)


# print out playbook_clause_name -> nda_clause_idx or None
print("playbook_clause_name -> nda_clause_idx or None")
print(playbook_to_nda_idx)



# --- Step 2: Redline detection and fix suggestion ---
redlined = []

for pb_name, nda_idx in playbook_to_nda_idx.items():
    if nda_idx is None:
        continue  # No NDA clause matched for this playbook clause type

    clause = bad_clauses[nda_idx]
    clause_content = clause.get('clause_content', '[NO CONTENT]')
    clause_heading = clause.get('clause_name', '[NO NAME]')

    #pb_record = next((pb for pb in playbook if pb.get("clause") == pb_name), None)

    #--------------------------------
    # Extracts and normalized playbook clause name
    def normalize_name(name):
        # Remove leading dash, whitespace, lowercase, collapse spaces
        return name.lstrip("-").strip().lower().replace("’", "'").replace("‘", "'")

    # Build a normalized lookup dict for playbook clause names
    playbook_lookup = {normalize_name(pb.get("clause", "")): pb for pb in playbook}

    # When processing mapping:
    pb_norm = normalize_name(pb_name)
    pb_record = playbook_lookup.get(pb_norm, None)
    if not pb_record:
        print(f"WARNING: Playbook entry for '{pb_name}' not found.")
        continue
    
    #--------------------------------


    if not pb_record:
        print(f"WARNING: Playbook entry for {pb_name} not found.")
        continue

    # --- Extract playbook fields for acceptability check ---
    clause_definition    = pb_record.get("clause_definition", "")
    provision_definition = pb_record.get("provision_definition", "")
    review_instruction   = pb_record.get("review_instruction", "")
    ideal                = pb_record.get("ideal", "")
    acceptable           = pb_record.get("acceptable", "")
    red_flags            = pb_record.get("red_flag", "")

    # --- Acceptability LLM prompt ---
    accept_prompt = f"""
You are a legal contract reviewer.

Below is an NDA clause, and the playbook guidance for its type.

- Begin by carefully reviewing the "Clause Definition" and "Provision Definition" to understand the intended purpose and legal context.
- Apply the "Review Instructions" as a checklist—these are the essential requirements, best practices, and explicit red flags.
- Refer to the "Ideal" and "Acceptable" examples: Acceptable is the minimum standard; Ideal is best practice.
- Pay special attention to any "Red Flags" listed.

CRITICAL INSTRUCTIONS:
- A clause is PROBLEMATIC if it omits any essential requirement listed in the Review Instructions, or contains any red flag language (such as permitting disclosure without prior notice, failing to require efforts to limit disclosure, waiving liability for breach of confidentiality, or precluding claims for damages/injunctive relief).
- A clause is ACCEPTABLE if, in your expert judgment, it offers at least the same level of protection as the Acceptable Example, does not introduce any clear new risks or red flag language, and substantially addresses the Review Instructions' core requirements—even if some minor details are omitted, sequence differs, or the language is not identical.
- Do not accept a clause merely because it resembles the Acceptable Example in some respects; all core requirements and red flags must be checked.
- If the clause omits only best-practice (Ideal) points but meets all Acceptable requirements and avoids all red flags, answer YES.
- Only answer NO if the clause is clearly weaker, more permissive, or introduces a real risk not present in the Acceptable Example.

For transparency, briefly explain which checklist items were met or missed, and call out any red flags found.

--- Playbook Guidance ---

Clause Definition:
{clause_definition}

Provision Definition:
{provision_definition}

Review Instructions:
{review_instruction}

Ideal Example:
{ideal}

Acceptable Example:
{acceptable}

Red Flags:
{red_flags}

--- NDA Clause to Review ---

{clause_content}

Explain in 1-3 sentences, then answer YES or NO on a new line. Do not write anything else after YES or NO.

"""

    accept_response = openai.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": "You are a legal contract reviewer."},
            {"role": "user", "content": accept_prompt}
        ],
        max_tokens=256,
        temperature=0
    )
    accept_result = accept_response.choices[0].message.content.strip()
    lines = [l.strip() for l in accept_result.splitlines() if l.strip()]
    yesno_line = lines[-1].upper() if lines else ""
    is_acceptable = yesno_line == "YES"
    explanation = "\n".join(lines[:-1]) if len(lines) > 1 else ""

    if not is_acceptable:
        print(f"\n--- LLM Explanation for Playbook Clause '{pb_name}' (NDA clause {nda_idx+1}) ---\n{explanation}\n")

    if is_acceptable:
        print(f"Playbook clause '{pb_name}' (NDA clause {nda_idx+1}): ACCEPTABLE (LLM) — Skipping redline.")
        continue  # Do not redline

    # --- Otherwise, problematic: get ideal/fallback fix from playbook ---
    fix = pb_record.get("example_ideal_clause") or pb_record.get("example_fallback_clause") or "[NO FIX AVAILABLE]"

    # --- LLM-based numbering/formatting harmonization for fix ---
    format_prompt = f"""
Original NDA clause:
{clause_content}

Replacement clause from the playbook:
{fix}

When suggesting the replacement clause, format its numbering, lettering, and structure to match the original clause as closely as possible. Do not change the legal substance—only the numbering and formatting. Output only the rewritten clause.
"""
    format_response = openai.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": "You are an expert legal drafter."},
            {"role": "user", "content": format_prompt}
        ],
        max_tokens=512,
        temperature=0
    )
    formatted_fix = format_response.choices[0].message.content.strip()

    output = {
        "text_snippet": clause_content,
        "playbook_clause_reference": pb_name,
        "suggested_fix": formatted_fix
    }
    redlined.append(output)
    print(f"Playbook clause '{pb_name}' (NDA clause {nda_idx+1}): PROBLEMATIC — Added to redlined output.")

# --- Print all redlined results ---
print("\nRedlined issues found:")
for item in redlined:
    print(json.dumps(item, indent=2) + "\n")

# --- Save all redline results to file
#timestamp = ... # e.g., datetime.now().strftime("%Y%m%d_%H%M%S")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
with open(f"redlined_output_{timestamp}.json", "w", encoding="utf-8") as f:
    json.dump(redlined, f, indent=2)
print(f"\nSaved redlined issues to: redlined_output_{timestamp}.json")

