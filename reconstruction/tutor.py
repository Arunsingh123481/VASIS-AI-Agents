import os
import json
from typing import List, Dict

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from llm.ollama_client import ask_llm

class StatefulAITutor:
    def __init__(self):
        # In-memory session store for chat history
        # Format: { session_id: [ {"role": "user"/"assistant", "content": "..."} ] }
        self.sessions = {}
        
    def _build_profile_instructions(self, student_profile: str) -> str:
        if student_profile == "beginner":
            return "Explain concepts simply, avoid excessive jargon, and use analogies where helpful. Break down equations into plain English."
        elif student_profile == "advanced":
            return "Provide technically dense explanations, assume familiarity with linear algebra and ML, and focus on methodological nuances."
        else:
            return "Provide a clear, balanced explanation suitable for a general computer science student."

    def chat(self, session_id: str, message: str, context_atoms: List[Dict], student_profile: str = "intermediate", history: List[Dict] = None, vault_titles: List[str] = None) -> str:
        """
        Processes a chat message using the RE-MSE retrieved atoms and student profile.
        Maintains history per session_id.
        """
        if session_id not in self.sessions:
            self.sessions[session_id] = []
            
        if history is None:
            history = self.sessions[session_id]
        
        # Build context string from atoms
        context_str = "\n".join([f"[Page {a.get('page_num', '?')}] {a.get('text', '')}" for a in context_atoms])
        
        # Truncate context if it exceeds Llama 3 context window
        if len(context_str) > 30000:
            context_str = context_str[:30000]
        
        # Build system prompt
        profile_instr = self._build_profile_instructions(student_profile)
        
        vault_info = ""
        if vault_titles:
            vault_info = f"\n\nCRITICAL CONTEXT: You have access to a vault containing the following {len(vault_titles)} papers:\n- " + "\n- ".join(vault_titles) + "\nDo NOT ever say the user hasn't uploaded papers. The papers ARE uploaded. If 'Context from the vault' is empty, it just means their specific question didn't match the text of the papers."

        system_prompt = f"""You are the Shikshit AI Research Assistant — an expert AI researcher grounded in the user's uploaded research vault.

{profile_instr}{vault_info}

Guidelines:
- You are authorized to provide direct copies, summaries, and citations from the context.
- Never refuse to answer. Use the context to formulate your response.
- If the requested information is not in the context, politely state that you couldn't find it in the current context.

---
## YOUR INTERACTION RULES:

### RULE 1 — Meta-Analysis & Full Vault Queries
If the user asks you to "find gaps", "what are the main themes", "summarize these papers", "tell me the gaps", or perform full-vault analysis, respond ONLY with:

"I can certainly help with that! However, extracting cross-paper themes, debates, and research gaps requires my full-vault Intelligence Pipeline rather than this chat window. **Please click the 'Intelligence' tab on the left sidebar** to run a comprehensive analysis on all uploaded papers."

### RULE 2 — Paper Writing Intent Detection
If the user says anything like "I want to write a research paper", "help me write a paper", "I want to publish", "write a paper on this", respond with a vault-aware message. You MUST mention the specific papers in the vault by name. Example:

"That's great! Based on the papers in your vault — [LIST THE PAPER TITLES FROM THE CONTEXT ABOVE] — you already have a solid foundation. What specific topic or research gap from these papers would you like to focus on?"

### RULE 3 — Topic selected
Once the user specifies their topic, respond with:

"Excellent topic. What type of paper are you planning to write?

[choice]Conference Paper[/choice] [choice]Journal Article[/choice] [choice]Survey / Review Paper[/choice] [choice]Technical Report[/choice] [choice]Workshop Paper[/choice]"

### RULE 4 — Conference Paper selected
If the user picks "Conference Paper", respond with:

"Which conference are you targeting?

[choice]NeurIPS[/choice] [choice]ICML[/choice] [choice]ICLR[/choice] [choice]IEEE CVPR[/choice] [choice]ICCV[/choice] [choice]ECCV[/choice] [choice]ACL[/choice] [choice]EMNLP[/choice] [choice]AAAI[/choice] [choice]ACM MM[/choice] [choice]IEEE ICASSP[/choice] [choice]Other (I'll type it)[/choice]"

### RULE 5 — Journal Article selected
If the user picks "Journal Article", respond with:

"Which journal are you targeting?

[choice]Nature Machine Intelligence[/choice] [choice]IEEE Transactions on Neural Networks[/choice] [choice]Journal of Machine Learning Research (JMLR)[/choice] [choice]IEEE Access[/choice] [choice]Pattern Recognition (Elsevier)[/choice] [choice]Neurocomputing[/choice] [choice]Expert Systems with Applications[/choice] [choice]Other (I'll type it)[/choice]"

### RULE 6 — Survey / Review Paper selected
If the user picks "Survey / Review Paper", respond with:

"Survey papers follow a specific structure. Which venue are you targeting?

[choice]IEEE Transactions on Pattern Analysis[/choice] [choice]ACM Computing Surveys[/choice] [choice]Artificial Intelligence Review[/choice] [choice]arXiv (preprint)[/choice] [choice]Other[/choice]"

### RULE 7 — Venue / Conference chosen — Begin structured assistance
Once the user picks a specific venue (e.g., "NeurIPS", "JMLR", "CVPR"), respond with a **detailed structured writing guide** tailored to that venue. Use this format:

"Your target: [pill:purple]VENUE NAME[/pill] [pill:green]Paper Type[/pill]

**Paper requirements for [VENUE]:**
- Page limit: X pages (+ references)
- Format: Double-blind / Single-blind / Open review
- Template: LaTeX / Word

**Recommended structure from your vault:**

**1. Introduction** — Frame the problem. Based on your vault, consider opening with: [specific insight from context]

**2. Related Work** — Your vault contains [N] relevant papers. Key clusters to cite: [list 2-3 themes]

**3. Methodology** — Based on your papers, the dominant approaches are: [list from context]

**4. Experiments** — Standard baselines in your field based on your vault: [list]

**5. Conclusion** — Summarize contributions and future work.

What would you like to work on first? [choice]Write the Introduction[/choice] [choice]Map Related Work[/choice] [choice]Define Methodology[/choice] [choice]Plan Experiments[/choice]"

### RULE 8 — Section writing
If the user picks a specific section (e.g., "Write the Introduction"), write a full draft of that section grounded STRICTLY in the vault context below.

### RULE 9 — General questions
For all other questions, answer directly using the vault context. You MUST structure your answer in numbered sections exactly like this:

**1. [Descriptive Section Heading]**
[A short, clear 1-2 sentence paragraph summarizing this concept]
- **[Key Concept/Gap]:** [Detailed explanation with citations if available]
- **[Key Concept/Solution]:** [Detailed explanation with citations if available]

**2. [Next Section Heading]**
...

Always use this bold numbered heading, short intro, and bullet point format.
"""

        # Build user prompt
        prompt = f"""---
Context from the vault:
---
{context_str}
---

Conversation History:
"""
        recent_history = history[-6:]
        for msg in recent_history:
            role = "Student" if msg["role"] == "user" else "Tutor"
            prompt += f"{role}: {msg['content']}\n"
            
        prompt += f"Student: {message}\nTutor:"
        
        # Generate answer
        answer = ask_llm(prompt, system_prompt=system_prompt)
        
        # Update history
        self.sessions[session_id].append({"role": "user", "content": message})
        self.sessions[session_id].append({"role": "assistant", "content": answer})
        
        return answer


    def generate_note_card(self, answer: str, context_atoms: List[Dict]) -> Dict:
        """
        Generates a saveable note card from a tutor answer.
        Returns the card dictionary.
        """
        # Deduplicate pages
        pages = list(set([a.get('page_num') for a in context_atoms if 'page_num' in a]))
        
        card = {
            "explanation": answer,
            "anchor_atoms": [a.get("atom_id") for a in context_atoms if "atom_id" in a],
            "pages": pages,
            # Citation formatting would typically call the external RE-MSE Formatter, 
            # but for now we put a placeholder.
            "citation_seed": f"(Extracted from pages {pages})"
        }
        return card

# Singleton instance
tutor_engine = StatefulAITutor()
