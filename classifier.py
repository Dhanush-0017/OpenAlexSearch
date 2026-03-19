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
    system = """You are an expert Computer Science and AI/ML research analyst.
Classify research papers into exactly one of three categories based on PRIMARY contribution.

CRITICAL RULE — AI AS SUBJECT, NOT TOOL:
The paper must be ABOUT improving AI/ML models themselves.
If AI/ML is only used as a tool for another domain (biology, chemistry, medicine, physics, finance), classify as "neither".

CATEGORIES:
- "efficiency" -> PRIMARY goal is making AI/ML models faster, smaller, cheaper, or less energy-intensive.
  Signals: model compression, pruning, quantization, knowledge distillation, sparse models, early exit, efficient inference.

- "scaling"    -> PRIMARY goal is improving accuracy, capability, or benchmark performance of AI/ML models.
  Signals: new architectures with SOTA results, larger models, better training strategies, improved objectives.

- "neither"    -> Paper is not about AI/ML, uses AI as a tool for another domain, or focuses on
  interpretability, fairness, alignment, safety, RLHF, datasets, surveys, or security.

TIEBREAKER: If paper claims BOTH efficiency AND accuracy gains, classify by what problem it was WRITTEN TO SOLVE.
BOUNDARY CASE — Mixture of Experts (MoE): if MoE is used to reduce compute -> efficiency. If used to improve capability -> scaling.

EXAMPLES (with reasoning):
- Title: "Pruned BERT runs 3x faster with minimal accuracy loss"
  Primary contribution: reduces model size and inference time via pruning
  Classification: efficiency — written to solve a speed/cost problem

- Title: "New transformer architecture achieves state-of-the-art on GLUE benchmark"
  Primary contribution: new architecture that improves task accuracy
  Classification: scaling — written to solve an accuracy/capability problem

- Title: "Scaling to 1 trillion parameters shows emergent capabilities"
  Primary contribution: trains a larger model, discovers new capabilities
  Classification: scaling — written to push model capability further

- Title: "Quantized LLM matches full-precision accuracy at 4-bit"
  Primary contribution: reduces memory and compute via quantization
  Classification: efficiency — written to make the model cheaper to run

- Title: "LLMs used to predict drug transfection efficiency in biology"
  Primary contribution: applies LLMs to a biology problem
  Classification: neither — AI is a tool, the paper is about biology

- Title: "Why large language models hallucinate — detection method proposed"
  Primary contribution: studies and detects a failure mode of LLMs
  Classification: neither — focuses on reliability/safety, not efficiency or scaling

- Title: "RLHF fine-tuning makes LLMs follow instructions better"
  Primary contribution: aligns model behavior with human preferences
  Classification: neither — alignment/safety, not efficiency or scaling"""

    prompt = f"""First identify the PRIMARY contribution of this paper, then classify it.

Title    : {title}
Journal  : {journal}
Concepts : {concepts}
Keywords : {keywords}
Abstract : {abstract[:600]}

Reply in this exact JSON format only — no extra text, no markdown:
{{
  "classification": "efficiency" or "scaling" or "neither",
  "accuracy": a number from 0 to 100 representing how confident you are,
  "reason": "one sentence explaining the primary contribution and why it fits the chosen category"
}}"""

    payload = {
        "model"      : ollama_model,
        "system"     : system,   #instructions stay anchored separately from paper data
        "prompt"     : prompt,
        "stream"     : False,
        "temperature": 0,        #deterministic — same result every run
        "num_predict": 400,      #cap response length to prevent rambling
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
