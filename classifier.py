import requests
import json


def _parse_ollama_response(raw):
    #strip markdown code blocks ollama sometimes adds
    raw = raw.replace("```json", "").replace("```", "").strip()

    #raises json.JSONDecodeError if response is not valid json
    result = json.loads(raw)

    #normalize classification to lowercase
    if "classification" in result:
        result["classification"] = result["classification"].strip().lower()

    #reject unexpected labels
    valid_labels = ["efficiency", "scaling", "neither"]
    if result.get("classification") not in valid_labels:
        result["classification"] = "unclear"

    #validate accuracy is a number between 0 and 100
    try:
        accuracy = int(result.get("accuracy", 0))
        result["accuracy"] = max(0, min(100, accuracy))
    except (ValueError, TypeError):
        result["accuracy"] = 0

    #ensure reason is always present
    if not result.get("reason"):
        result["reason"] = "no reason provided"

    return result


def classify_paper(title, abstract, ollama_url, ollama_model, journal="N/A", concepts="N/A", keywords="N/A"):
    system = """You are an expert AI/ML research paper classifier.

Your job is to classify a paper into one of three categories: "efficiency", "scaling", or "neither".

STEP 1 — Ask yourself: Does this paper propose a new AI/ML method, model, or training technique?
  - If YES → go to Step 2
  - If NO (e.g. applies existing AI to biology, chemistry, medicine with no new AI contribution) → "neither"

STEP 2 — Ask yourself: What is the PRIMARY goal of the new method?
  - Goal is to make AI models FASTER, SMALLER, or CHEAPER to run → "efficiency"
    (pruning, quantization, distillation, LoRA, sparse models, efficient attention, early exit, reduced memory)
  - Goal is to make AI models MORE ACCURATE, MORE CAPABLE, or BETTER at tasks → "scaling"
    (new architectures, better training strategies, RLHF, instruction tuning, contrastive learning,
     self-supervised learning, new objectives, multi-modal models, better generalization, SOTA results)

IMPORTANT RULES:
- "neither" is a last resort. Only use it when the paper's main contribution is clearly outside AI/ML.
- If a paper improves BOTH speed and accuracy, classify by the PROBLEM IT WAS WRITTEN TO SOLVE.
- RLHF, instruction tuning, alignment, and fine-tuning strategies → "scaling" (they improve model quality).
- Interpretability or safety methods that also improve model performance → "scaling".

EXAMPLES:
Title: "LoRA: Low-Rank Adaptation of Large Language Models"
Reasoning: Proposes a new fine-tuning method that reduces trainable parameters → makes training cheaper
Classification: efficiency

Title: "FlashAttention: Fast and Memory-Efficient Exact Attention"
Reasoning: Proposes a faster attention algorithm → reduces memory and speeds up training
Classification: efficiency

Title: "Scaling to 1 trillion parameters shows emergent capabilities"
Reasoning: Trains a much larger model and studies new capabilities → improves model capability
Classification: scaling

Title: "RLHF fine-tuning makes LLMs follow instructions better"
Reasoning: Proposes a training strategy to improve instruction-following quality → better model capability
Classification: scaling

Title: "Instruction tuning improves zero-shot generalization"
Reasoning: New training approach to improve generalization across tasks → better model quality
Classification: scaling

Title: "LLMs used to predict drug-protein interactions"
Reasoning: Uses existing LLMs as a tool for a biology problem, no new AI method proposed
Classification: neither"""

    prompt = f"""Read the paper details below and classify it step by step.

Title    : {title}
Journal  : {journal}
Concepts : {concepts}
Keywords : {keywords}
Abstract : {abstract[:800]}

Think through these questions before answering:
1. What is the main new thing this paper proposes? (a new AI method, or applying AI to another field?)
2. If it proposes a new AI method — is the goal to make models faster/smaller, or more accurate/capable?

Reply in this exact JSON format only — no extra text, no markdown:
{{
  "reason": "one sentence: what the paper proposes and why it fits the chosen category",
  "classification": "efficiency" or "scaling" or "neither",
  "accuracy": a number from 0 to 100 representing how confident you are
}}"""

    payload = {
        "model"      : ollama_model,
        "system"     : system,   #instructions stay anchored separately from paper data
        "prompt"     : prompt,
        "stream"     : False,
        "temperature": 0,        #deterministic — same result every run
        "num_predict": 300,      #reason + classification + accuracy fits in 300 tokens
    }

    try:
        #retry once if ollama returns bad json
        for attempt in range(2):
            try:
                response = requests.post(ollama_url, json=payload, timeout=60)
                raw = response.json()["response"].strip()
                return _parse_ollama_response(raw)

            except json.JSONDecodeError:
                if attempt == 0:
                    print("Ollama returned unexpected format. Retrying...")
                    continue
                #second attempt also failed
                print("Ollama gave an unexpected response after retry. Skipping this paper.")
                return {
                    "classification": "unclear",
                    "accuracy"      : 0,
                    "reason"        : "could not parse ollama response after retry"
                }

    except requests.exceptions.ConnectionError:
        #
        # ollama is not running
        print("Could not connect to Ollama. Make sure Ollama is running.")
        return {
            "classification": "error",
            "accuracy"      : 0,
            "reason"        : "could not connect to ollama"
        }

    except Exception as e:
        #anything else that goes wrong
        print(f"Something went wrong: {e}")
        return {
            "classification": "error",
            "accuracy"      : 0,
            "reason"        : str(e)
        }
