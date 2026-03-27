import re
import json
import requests
import os

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


def summarize_and_classify(title, combined_text):
    prompt = f"""You are analyzing a research paper titled: "{title}"

Below is the content of the paper:

{combined_text}

Complete TWO tasks:

TASK 1 - SUMMARY:
Write a concise summary in 200-300 words. Use plain text only, no markdown, no bullet points, no bold or italic formatting.
Cover: the problem being solved, the methodology used, and the key contributions.

TASK 2 - CLASSIFICATION:
Classify this paper into one of two categories: [Efficiency] or [Scaling].

Category A: Efficiency
Focus: Doing more with less. Reducing the computational footprint of a model without necessarily increasing its size.
Key Indicators: Quantization (int8, FP4), Pruning, Distillation, Low-rank Adaptation (LoRA), Sparse Attention mechanisms, Memory-efficient kernels (FlashAttention), or Reducing FLOPs for the same parameter count.
Goal: Inference speed, training cost reduction, or hardware constraints.

Category B: Scaling
Focus: Understanding or implementing the effects of increasing model size, data volume, or compute.
Key Indicators: Scaling Laws (Chinchilla, Kaplan), Emergent properties, Mixture of Experts (MoE) at scale, High-parameter counts (70B+), or "Data-constrained" scaling.
Goal: Improving general intelligence, capability benchmarks, or studying behavior at the limit.

Instructions:
- Determine which category is the "Primary Contribution." If a paper scales an efficient method, classify it by the method being introduced.
- Provide a Confidence Score (0-100).
- Provide a detailed Justification of 3-4 sentences focusing on the technical methodology, key techniques used, and why this paper fits the chosen category over the other.

Reply in this EXACT format (no extra text):
Summary:
<your 200-300 word summary here>

Category: <Efficiency or Scaling>
Confidence: <0-100>
Justification: <3-4 sentences focusing on the technical methodology>"""

    return call_llm(prompt, timeout=300)


def parse_response(raw):
    summary       = ""
    category      = "Unknown"
    confidence    = "0"
    justification = ""

    # Extract summary block (between "Summary:" and "Category:")
    summary_match = re.search(r'Summary:\s*\n(.*?)(?=\nCategory:)', raw, re.DOTALL)
    if summary_match:
        summary = summary_match.group(1).strip()

    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("Category:"):
            category      = line.split(":", 1)[1].strip()
        elif line.startswith("Confidence:"):
            confidence    = line.split(":", 1)[1].strip()
        elif line.startswith("Justification:"):
            justification = line.split(":", 1)[1].strip()

    return summary, {"category": category, "confidence": confidence, "justification": justification}


def save_paper(data):
    file_path = os.path.join(OUTPUT_DIR, f"{data['paper_id']}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def process_paper(paper_id):
    data          = load_chunks(paper_id)
    combined_text = build_combined_text(data["chunks"])
    raw           = summarize_and_classify(data["title"], combined_text)
    summary, classification = parse_response(raw)

    data["summary"]        = summary
    data["classification"] = classification
    save_paper(data)
    return data
