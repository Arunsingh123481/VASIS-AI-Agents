# Guide to Customizing Models and Integrating External APIs in VASIS AI

This guide explains how to:
1. Switch to a different local **Ollama** model.
2. Integrate a cloud-based **LLM API** (such as OpenAI GPT, Anthropic Claude, or Google Gemini) using an API Key.

---

## 🗺️ Codebase Map for LLM Configuration

The core files responsible for LLM configuration, routing, and inference are:
1. **[config.py](file:///e:/Vasis%20AI/config.py)**: Central configuration for model names, ports, and API parameters.
2. **[.env](file:///e:/Vasis%20AI/.env)**: Environment variables file where API keys are stored.
3. **[llm/ollama_client.py](file:///e:/Vasis%20AI/llm/ollama_client.py)**: The wrapper that sends prompts to the LLM backend (currently local Ollama).
4. **[llm/router.py](file:///e:/Vasis%20AI/llm/router.py)**: Determines whether a request gets sent to the Agentic model or Reasoning model.
5. **[pipeline.py](file:///e:/Vasis%20AI/pipeline.py)**: Performs pre-flight health checks to ensure the LLM service is available before starting ingestion.

---

## 1. How to Change Local Ollama Models

If you want to use a different or larger local model (e.g., `qwen2.5-coder:7b`, `llama3.1:8b`, `deepseek-coder:6.7b`), follow these steps:

### Step 1: Pull the model on your system
Run the pull command in your terminal/powershell:
```powershell
ollama pull qwen2.5-coder:7b
```

### Step 2: Update model names in `config.py`
Open **[config.py](file:///e:/Vasis%20AI/config.py)** and update the model names at lines 11–15:
```python
# Local models downloaded by user
AGENT_MODEL = "qwen2.5-coder:7b"          # Updated from 3b to 7b
REASONING_MODEL = "deepseek-llm:7b"      # Change this if you pull a larger reasoning model

# Fallbacks
DEFAULT_MODEL = "qwen2.5-coder:7b"
```

---

## 2. How to Integrate Cloud APIs (OpenAI, Claude, Gemini, etc.)

To swap the local inference engine with a larger cloud model (such as GPT-4o, Claude 3.5 Sonnet, or Gemini 1.5 Pro) using an API key:

### Step 1: Add the API Key to `.env`
Add your API key at the end of the **[.env](file:///e:/Vasis%20AI/.env)** file:
```ini
OPENAI_API_KEY=sk-proj-yourActualOpenAIApiKeyHere
# or
ANTHROPIC_API_KEY=sk-ant-yourActualAnthropicApiKeyHere
```

### Step 2: Load the key in `config.py`
Open **[config.py](file:///e:/Vasis%20AI/config.py)** and add the environment variable retrieval near line 57 (under Agent 12 variables):
```python
# Cloud API Keys
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
```

### Step 3: Install necessary client SDKs
Add the client library you want to use to **[requirements.txt](file:///e:/Vasis%20AI/requirements.txt)**:
```txt
openai>=1.0.0
# or
anthropic>=0.18.0
```
Then run:
```powershell
pip install -r requirements.txt
```

### Step 4: Modify `llm/ollama_client.py` to route to Cloud API
Open **[llm/ollama_client.py](file:///e:/Vasis%20AI/llm/ollama_client.py)**. Modify `ask_llm()` to check if the model name indicates a cloud model (like starting with `gpt-` or `claude-`) and make an API request if so.

Here is an example implementation using the **OpenAI API**:

```python
# Import the cloud SDKs at the top of llm/ollama_client.py
from openai import OpenAI
from config import OPENAI_API_KEY

# Initialize cloud client if API key is provided
openai_client = None
if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
```

Then update `ask_llm()` to route requests dynamically:

```python
def ask_llm(
    prompt: str,
    model: str = None,
    retries: int = API_RETRIES,
    expect_json: bool = False,
    system_prompt: str = None,
    temperature: float = 0.0
) -> str:
    """
    Send a prompt to either OpenAI or the local Ollama server depending on the model name.
    """
    # 1. Fallback & Auto-routing configuration logic
    if model in (None, "llama3.2", DEFAULT_MODEL):
        # Your default logic or router override:
        # e.g., if you want cloud to be the default: model = "gpt-4o"
        combined = ((system_prompt or "") + " " + prompt).lower()
        reasoning_keywords = [
            "tutor", "student", "explain", "lesson", "hallucination", 
            "audit", "grounding", "contradiction", "novelty", "hypothesis", 
            "debate", "evidence", "claim", "synthesis", "answer", "respond"
        ]
        if any(keyword in combined for keyword in reasoning_keywords):
            model = "gpt-4o"  # Use cloud model for reasoning tasks
        else:
            model = AGENT_MODEL # Keep using local Ollama for fast agentic work

    # 2. Route to Cloud API if applicable
    if model.startswith("gpt-") and openai_client:
        for attempt in range(retries):
            try:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                chat_kwargs = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                }
                # Force JSON format if requested
                if expect_json:
                    chat_kwargs["response_format"] = { "type": "json_object" }

                response = openai_client.chat.completions.create(**chat_kwargs)
                content = response.choices[0].message.content.strip()
                
                if expect_json:
                    content = _extract_json(content)
                return content
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(2)
                else:
                    raise ModelError(f"OpenAI API failed: {e}")

    # 3. Otherwise, fallback to the local Ollama client implementation
    # (Existing Ollama execution block goes here...)
```

### Step 5: Handle Pre-Flight Connection Checks
Since external APIs don't need a local Ollama connection, update the health check in **[llm/ollama_client.py](file:///e:/Vasis%20AI/llm/ollama_client.py)** to bypass checks for cloud models:

```python
def check_ollama_connection(model: str = DEFAULT_MODEL) -> bool:
    """Check if model service is available (returns True immediately for cloud models)."""
    if model.startswith("gpt-") or model.startswith("claude-") or model.startswith("gemini-"):
        # Cloud API is handled via internet connection; skip local server check
        return True
        
    # Existing local Ollama checks...
```

This ensures **[pipeline.py](file:///e:/Vasis%20AI/pipeline.py)** and **[api.py](file:///e:/Vasis%20AI/api.py)** won't throw errors when initializing the system.
