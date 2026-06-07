"""
Ollama Client — Wrapper for Llama, Qwen, and DeepSeek via Ollama local server.
Handles all local LLM calls with retry logic, robust JSON parsing, and error handling.
"""
import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))

import json
import time
import requests
import re
from config import OLLAMA_URL, DEFAULT_MODEL, AGENT_MODEL, API_TIMEOUT, API_RETRIES
from utils.exceptions import ModelError
from console_helper import print_msg

# Set OLLAMA_HOST environment variable before importing ollama
_os.environ["OLLAMA_HOST"] = OLLAMA_URL

import ollama

# Create client with generous timeout
_client = ollama.Client(host=OLLAMA_URL, timeout=API_TIMEOUT)


def ask_llm(
    prompt: str,
    model: str = None,
    retries: int = API_RETRIES,
    expect_json: bool = False,
    system_prompt: str = None,
    temperature: float = 0.0
) -> str:
    """
    Send a prompt to local Ollama server and return the response text.
    Intelligently routes to Qwen or DeepSeek based on task keywords if not explicitly specified.
    """
    if model in (None, "llama3.2", DEFAULT_MODEL):
        combined = ((system_prompt or "") + " " + prompt).lower()
        reasoning_keywords = [
            "tutor", "student", "explain", "lesson", "hallucination", 
            "audit", "grounding", "contradiction", "novelty", "hypothesis", 
            "debate", "evidence", "claim", "synthesis", "answer", "respond"
        ]
        if any(keyword in combined for keyword in reasoning_keywords):
            model = "deepseek-llm:7b"
        else:
            model = AGENT_MODEL

    for attempt in range(retries):
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            chat_kwargs = {
                "model": model,
                "messages": messages,
                "options": {"temperature": temperature, "num_predict": 4096}
            }
            if expect_json:
                chat_kwargs["format"] = "json"

            response = _client.chat(**chat_kwargs)
            content = response['message']['content'].strip()
            if not content:
                raise ValueError("Ollama returned an empty response")

            if expect_json:
                content = _extract_json(content)

            return content

        except Exception as e:
            if attempt < retries - 1:
                print_msg(f"[yellow]Ollama call ({model}) failed (attempt {attempt+1}/{retries}): {e}. Retrying...[/yellow]")
                time.sleep(2)
            else:
                print_msg(f"[red]Ollama call ({model}) failed after {retries} attempts: {e}[/red]")
                if expect_json:
                    return "{}"
                raise ModelError(f"Ollama failed to generate response: {e}")

    return ""


def _extract_json(text: str) -> str:
    """
    Extract JSON from LLM response that may contain markdown fences or conversational text.
    Handles trailing commas and single-line comments gracefully.
    """
    # Clean up markdown code blocks if any
    raw = re.sub(r"```json\s*", "", text)
    raw = re.sub(r"```\s*", "", raw).strip()

    # Try direct parse first
    try:
        json.loads(raw)
        return raw
    except json.JSONDecodeError:
        pass

    # Regex search for brackets/braces
    cleaned = None
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start = raw.find(start_char)
        end = raw.rfind(end_char)
        if start != -1 and end != -1 and end > start:
            candidate = raw[start:end+1]
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                # Try repairing the candidate block
                repaired = _repair_json_string(candidate)
                try:
                    json.loads(repaired)
                    return repaired
                except json.JSONDecodeError:
                    cleaned = candidate
                    continue

    # Try repairing the full target
    target = cleaned if cleaned else raw
    repaired = _repair_json_string(target)
    try:
        json.loads(repaired)
        return repaired
    except json.JSONDecodeError:
        pass

    return text


def _repair_json_string(text: str) -> str:
    """Helper to remove trailing commas and comments from LLM-generated JSON."""
    # Remove single-line comments starting with // but not inside URLs
    text = re.sub(r'(?<!:)\/\/.*$', '', text, flags=re.MULTILINE)
    # Remove trailing commas before closing braces/brackets
    text = re.sub(r',\s*([\}\]])', r'\1', text)
    return text.strip()


def check_ollama_connection(model: str = DEFAULT_MODEL) -> bool:
    """Check if Ollama is running and the specified model is pulled and available."""
    try:
        result = _client.list()
        if hasattr(result, 'models'):
            available = [m.model if hasattr(m, 'model') else str(m) for m in result.models]
        elif isinstance(result, dict):
            available = [m.get('name', str(m)) for m in result.get('models', [])]
        else:
            available = []
            
        if not any(model in m for m in available):
            print_msg(f"[yellow]Model '{model}' not found in local Ollama. Available: {available}[/yellow]")
            print_msg(f"[yellow]Run: ollama pull {model}[/yellow]")
            return False
        return True
    except Exception as e:
        print_msg(f"[red]Cannot connect to Ollama at {OLLAMA_URL}: {e}[/red]")
        print_msg("[red]Make sure Ollama is running: ollama serve[/red]")
        return False
