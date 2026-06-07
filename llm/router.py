# llm/router.py
# Routes agent calls to correct local model based on their operational nature

from config import AGENT_MODEL, REASONING_MODEL
import llm.ollama_client as ollama_client
from console_helper import print_msg

# Categorize agents to assign either Qwen (agentic) or DeepSeek (reasoning)
AGENTIC_WORK = {
    "agent1_router",
    "agent2_decomposer",
    "agent3_navigator",
    "agent8_temporal",
    "agent10_super",
    "agent12_websearch",
    "tree_builder",
    "triple_extractor",
}

REASONING_WORK = {
    "agent6_validation",
    "agent7_contradiction",
    "agent11_synthesis",
    "agent13_paper_writer",
    "agent14_impl_guide",
    "answer_generation",
    "tutor",
    "novelty",
}

def get_model(agent_name: str) -> str:
    """Returns Ollama model name for a given agent/task."""
    if agent_name in AGENTIC_WORK:
        return AGENT_MODEL
    elif agent_name in REASONING_WORK:
        return REASONING_MODEL
    return AGENT_MODEL  # Fallback to Qwen for unspecified agents


def generate(
    agent_name: str,
    prompt: str,
    system: str = None,
    temperature: float = 0.0
) -> str:
    """Route generate call to correct local Ollama model."""
    model = get_model(agent_name)
    # Check if local model is connected/available
    ollama_client.check_ollama_connection(model)
    
    print_msg(f"[Router] {agent_name} -> [bold cyan]{model}[/bold cyan] (generate)")
    return ollama_client.ask_llm(
        prompt=prompt,
        model=model,
        expect_json=False,
        system_prompt=system,
        temperature=temperature
    )


def generate_json(
    agent_name: str,
    prompt: str,
    system: str = None
) -> dict | list:
    """Route generate_json call to correct local Ollama model."""
    model = get_model(agent_name)
    # Check if local model is connected/available
    ollama_client.check_ollama_connection(model)
    
    print_msg(f"[Router] {agent_name} -> [bold magenta]{model}[/bold magenta] (generate_json)")
    raw = ollama_client.ask_llm(
        prompt=prompt,
        model=model,
        expect_json=True,
        system_prompt=system,
        temperature=0.0
    )
    import json
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print_msg(f"[red][Router] Failed to parse JSON response from {model}: {e}[/red]")
        # Graceful fallbacks
        if "synthesis" in agent_name or "contradiction" in agent_name or "decomposer" in agent_name:
            return []
        return {}
