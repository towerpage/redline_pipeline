import os
import re
import json
import sys

try:
    from striprtf.striprtf import rtf_to_text
except ImportError:
    rtf_to_text = None

def is_rtf_file(filepath):
    with open(filepath, 'rb') as f:
        header = f.read(64)
    return b'{\\rtf' in header

def load_text(filepath):
    if is_rtf_file(filepath):
        print("[INFO] Detected RTF format. Extracting text...")
        if rtf_to_text is None:
            print("[ERROR] 'striprtf' is required to process RTF files. Please install with: pip install striprtf")
            sys.exit(1)
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            rtf_content = f.read()
        text = rtf_to_text(rtf_content)
    else:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
    return text

CANONICAL_CLAUSE_NAMES = [
    "definitions", "confidentiality", "confidentiality obligations", "permitted and restricted uses", "permitted uses",
    "restricted uses", "term and termination", "term", "termination", "return or destruction of materials", "return of materials",
    "disclosures required by law", "required by law", "ownership", "ownership; no license", "no license", "indemnification",
    "assignment", "notices", "governing law", "oral agreements", "miscellaneous", "severability", "entire agreement",
    "amendment", "waiver", "counterparts", "remedies", "dispute resolution", "arbitration"
]
CANONICAL_CLAUSE_REGEX = re.compile(
    r"\b(" + "|".join([re.escape(name) for name in CANONICAL_CLAUSE_NAMES]) + r")\b",
    re.IGNORECASE
)

# Heuristics for detecting the start of a signature block
SIGNATURE_START_PATTERNS = [
    re.compile(r"^IN\s+WITNESS\s+WHEREOF", re.IGNORECASE),
    re.compile(r"^SIGNED\s+this", re.IGNORECASE),
    re.compile(r"^Executed", re.IGNORECASE),
    re.compile(r"^Signed for and on behalf", re.IGNORECASE),
    re.compile(r"^The parties (hereto|to this agreement)", re.IGNORECASE),
    re.compile(r"^Discloser[:：]\s*[_\s]*$", re.IGNORECASE),
    re.compile(r"^Recipient[:：]\s*[_\s]*$", re.IGNORECASE),
    re.compile(r"^Name[:：]\s*[_\s]*$", re.IGNORECASE),
    re.compile(r"^Title[:：]\s*[_\s]*$", re.IGNORECASE),
    re.compile(r"^Date[:：]\s*[_\s]*$", re.IGNORECASE),
    # Add more as needed for your organization
]

def is_signature_line(line):
    stripped = line.strip()
    for pat in SIGNATURE_START_PATTERNS:
        if pat.match(stripped):
            return True
    # Also consider lines that are mostly underscores or blank
    if re.match(r"^_{5,}$", stripped) or re.match(r"^\s*$", stripped):
        return True
    return False

def is_flexible_heading(line):
    stripped = line.strip()
    # ALL CAPS, not too long
    if (stripped.isupper() and 2 < len(stripped) < 80 and len(stripped.split()) < 10):
        return True
    # Heading with colon
    if re.match(r"^[A-Z][A-Za-z\s\-/;]+:$", stripped):
        return True
    # Canonical clause phrase, short line
    if CANONICAL_CLAUSE_REGEX.search(stripped) and len(stripped) < 80:
        return True
    return False

def is_numbered_heading(line):
    stripped = line.strip()
    # E.g., "2. Confidentiality", "10. Notices", "I. DEFINITIONS", "A. Permitted Uses"
    if re.match(r"^(\d+|[IVXLCDM]+|[A-Z])\.\s*[\w\(\[]+", stripped):
        if not re.match(r"^\d+\.\d", stripped):
            return True
    return False

def find_heading_indices(lines):
    headings = []
    for idx, line in enumerate(lines):
        if is_numbered_heading(line) or is_flexible_heading(line):
            headings.append((idx, line.strip()))
    # Deduplicate adjacent headings (only keep the first of each block)
    final_headings = []
    last_idx = -10
    for idx, text in headings:
        if idx - last_idx > 0:
            final_headings.append((idx, text))
        last_idx = idx
    return final_headings

def remove_preamble_headings(lines, heading_indices):
    preamble_patterns = [
        re.compile(r"non-?disclosure agreement", re.IGNORECASE),
        re.compile(r"this agreement (is|shall|constitutes|made|entered into)", re.IGNORECASE),
        re.compile(r"effective date", re.IGNORECASE),
        re.compile(r"by and between", re.IGNORECASE),
    ]
    # Remove first heading if preamble
    if heading_indices:
        idx, _ = heading_indices[0]
        block = "\n".join(lines[:idx+1])
        for pat in preamble_patterns:
            if pat.search(block):
                heading_indices = heading_indices[1:]
                break
    return heading_indices

def find_signature_block_start(lines, start_line):
    # Returns the line index of signature block start, or None
    for idx in range(start_line, len(lines)):
        if is_signature_line(lines[idx]):
            return idx
    return None

def extract_clauses(text):
    lines = text.splitlines()
    headings = find_heading_indices(lines)
    headings = remove_preamble_headings(lines, headings)
    if not headings:
        print("[WARN] No clause headings detected!")
        return []
    clause_ranges = []
    for i, (idx, heading) in enumerate(headings):
        start = idx
        # By default, end at the next clause heading
        if i+1 < len(headings):
            end = headings[i+1][0]
        else:
            # For final clause, check for signature block
            sig_start = find_signature_block_start(lines, start+1)
            end = sig_start if sig_start is not None else len(lines)
        clause_ranges.append((start, end, heading))
    clauses = []
    for start, end, heading in clause_ranges:
        block_lines = lines[start:end]
        clause_name = re.sub(r"[:：]+$", "", heading.strip())
        content_lines = block_lines[1:] if len(block_lines) > 1 else []
        clause_content = "\n".join([l.rstrip() for l in content_lines]).strip()
        if not clause_content and len(block_lines) > 1:
            clause_content = "\n".join(block_lines[1:]).strip()
        clauses.append({
            "clause_name": clause_name,
            "clause_content": clause_content
        })
    return clauses

def save_json(clauses, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(clauses, f, ensure_ascii=False, indent=2)

def main():
    if len(sys.argv) != 2:
        print("Usage: python extract_clauses.py <path_to_contract.txt>")
        sys.exit(1)
    filepath = sys.argv[1]
    if not os.path.isfile(filepath):
        print(f"[ERROR] File not found: {filepath}")
        sys.exit(1)
    text = load_text(filepath)
    clauses = extract_clauses(text)
    print(json.dumps(clauses, ensure_ascii=False, indent=2))
    print(f"\nTotal unique clauses detected: {len(clauses)}")
    output_basename = os.path.splitext(os.path.basename(filepath))[0] + "_clauses.json"
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_basename)
    save_json(clauses, output_path)
    print(f"\nSaved output to: {output_path}")

if __name__ == "__main__":
    main()
