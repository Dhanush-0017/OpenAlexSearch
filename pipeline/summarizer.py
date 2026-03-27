import re
import json
import requests
import os

from pipeline.classifier import CLASSIFICATION_PROMPT, parse_classification

# ── Constants ─────────────────────────────────────────────────────────────────
OLLAMA_URL   = "http://10.230.100.240:17020/api/generate"
OLLAMA_MODEL = "gpt-oss:120b"
OUTPUT_DIR   = "output"


def call_llm(prompt, timeout=300):
    response = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def load_chunks(paper_id):
    file_path = os.path.join(OUTPUT_DIR, f"{paper_id}.json")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def build_combined_text(chunks):
    skip     = {"references", "bibliography", "acknowledgements", "acknowledgments"}
    parts    = [c["text"] for c in chunks if c["section"] not in skip]
    combined = "\n\n".join(parts)
    return combined[:6000]


def build_prompt(title, combined_text):
    """Build the combined summarization + classification prompt."""
    return f"""You are analyzing a research paper titled: "{title}"

Below is the content of the paper:

{combined_text}

Complete TWO tasks:

TASK 1 - SUMMARY:
Write a concise summary in 200-300 words. Use plain text only, no markdown, no bullet points, no bold or italic formatting.
Cover: the problem being solved, the methodology used, and the key contributions.

{CLASSIFICATION_PROMPT}

Reply in this EXACT format (no extra text):
Summary:
<your 200-300 word summary here>

Category: <Efficiency or Scaling>
Confidence: <0-100>
Justification: <3-4 sentences focusing on the technical methodology>"""


def parse_summary(raw):
    """Parse the summary section from the LLM response."""
    match = re.search(r'Summary:\s*\n(.*?)(?=\nCategory:)', raw, re.DOTALL)
    return match.group(1).strip() if match else ""


def save_paper(data):
    file_path = os.path.join(OUTPUT_DIR, f"{data['paper_id']}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def process_paper(paper_id):
    data          = load_chunks(paper_id)
    combined_text = build_combined_text(data["chunks"])
    prompt        = build_prompt(data["title"], combined_text)
    raw           = call_llm(prompt)

    data["summary"]        = parse_summary(raw)
    data["classification"] = parse_classification(raw)
    save_paper(data)
    return data
