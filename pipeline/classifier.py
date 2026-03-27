import re

# ── Classification prompt section ─────────────────────────────────────────────
CLASSIFICATION_PROMPT = """TASK 2 - CLASSIFICATION:
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

Category: <Efficiency or Scaling>
Confidence: <0-100>
Justification: <3-4 sentences focusing on the technical methodology>"""


def parse_classification(raw):
    """Parse category, confidence, and justification from the LLM response."""
    result = {"category": "Unknown", "confidence": "0", "justification": ""}

    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("Category:"):
            result["category"]      = line.split(":", 1)[1].strip()
        elif line.startswith("Confidence:"):
            result["confidence"]    = line.split(":", 1)[1].strip()
        elif line.startswith("Justification:"):
            result["justification"] = line.split(":", 1)[1].strip()

    return result
