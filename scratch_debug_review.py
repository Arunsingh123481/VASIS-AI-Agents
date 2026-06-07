import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.agent10_super import _agent10_review
from agent_routing_rules import ROUTING_RULES
from agents.agent1_router import get_agents_for_query

def main():
    navigation = {
        "selected_nodes": [],
        "reasoning": "This section provides a detailed analysis...",
        "confidence": 0.8
    }
    routing = get_agents_for_query("Can you tell me the advantages and disadvantages of this paper?")
    review = _agent10_review("agent3_navigator", navigation, routing)
    print("REVIEW RESULT:", review)

if __name__ == "__main__":
    main()
