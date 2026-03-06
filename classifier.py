# this file sends the abstract to ollama and gets the classification

import requests
import json


def classify_paper(title, abstract, ollama_url, ollama_model):
    # i build a prompt that tells the llm what to do
    prompt = f"""Read the research paper title and abstract below.

Classify the paper as one of these:
- "efficiency" -> paper is about making models faster, smaller or cheaper
- "scaling"    -> paper is about making models bigger or using more data
- "both"       -> paper covers both efficiency and scaling
- "neither"    -> paper is about something else

Reply in this exact JSON format only, no extra text:
{{
  "classification": "efficiency" or "scaling" or "both" or "neither",
  "reason": "one sentence explaining why"
}}

Title: {title}
Abstract: {abstract}
"""

    try:
        # send the abstract to ollama
        response = requests.post(
            ollama_url,
            json={
                "model" : ollama_model,
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )

        # get the text response from ollama
        raw = response.json()["response"].strip()

        # ollama sometimes wraps response in ```json - remove it
        raw = raw.replace("```json", "").replace("```", "").strip()

        # try to parse the response as json
        result = json.loads(raw)

        # clean up classification - remove spaces and convert to lowercase
        # this handles cases like "Efficiency" or " scaling "
        if "classification" in result:
            result["classification"] = result["classification"].strip().lower()

        # check if classification is one of our expected labels
        valid_labels = ["efficiency", "scaling", "both", "neither"]
        if result.get("classification") not in valid_labels:
            result["classification"] = "unclear"

        # if ollama forgot to include a reason, add a default one
        if "reason" not in result or not result["reason"]:
            result["reason"] = "no reason provided"

        return result

    except requests.exceptions.ConnectionError:
        # ollama is not running
        print("Could not connect to Ollama. Make sure Ollama is running.")
        return {
            "classification": "error",
            "reason": "could not connect to ollama"
        }

    except json.JSONDecodeError:
        # ollama gave something we could not parse as json
        print("Ollama gave an unexpected response. Skipping this paper.")
        return {
            "classification": "unclear",
            "reason": "could not understand ollama response"
        }

    except Exception as e:
        # anything else that goes wrong
        print(f"Something went wrong: {e}")
        return {
            "classification": "error",
            "reason": str(e)
        }