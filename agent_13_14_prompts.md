# Agent 13 & Agent 14 Triggering Prompts

This guide lists the specific prompt patterns and keywords used to activate **Agent 13 (Research Paper Writer)** and **Agent 14 (Implementation Guide)** in the PageIndex-RE-MSE system.

Detection patterns are defined in [agent_routing_rules.py](file:///e:/Vasis%20AI/agent_routing_rules.py).

---

## 📄 Agent 13: Research Paper Writer

Agent 13 is triggered when the query indicates a request to write a paper or article. 

### Trigger Keywords/Phrases
Include any of the following phrases (case-insensitive) in your query:
* `write a paper`
* `write a research paper`
* `write paper`
* `draft a paper`
* `write a review paper`
* `write a review article`
* `write an article`
* `paper on`
* `research paper on`
* `review paper on`
* `short communication`
* `case study on`
* `perspective article`
* `technical note on`
* `systematic review on`
* `write for journal`
* `conference paper`
* `submit to`

### Example Prompts
* *“Write a research paper on the attention mechanism used in this document.”*
* *“Draft a paper summarizing the key contributions and results.”*
* *“I want to write a review article about the transformer architecture described here.”*

---

## 🔧 Agent 14: Implementation Guide

Agent 14 is triggered when the query requests code, implementation steps, or guides.

### Trigger Keywords/Phrases
Include any of the following phrases (case-insensitive) in your query:
* `how to implement`
* `implement this`
* `guide me`
* `show me how`
* `how do i build`
* `code for`
* `step by step`
* `implementation plan`
* `help me implement`
* `how to build`
* `development guide`
* `coding guide`
* `implementation guide`

### Example Prompts
* *“How to implement the multi-head self-attention module in PyTorch?”*
* *“Give me an implementation plan and code for training this model.”*
* *“Guide me step by step on building the position-wise feed-forward networks.”*

---

## 🧠 Joint Mode: Activating Both Agents

If your query contains keywords from **both** lists, the system enters a joint `deep_research` mode. Both Agent 13 and Agent 14 will run, and their outputs will be combined.

### Example Prompts
* *“Write a paper on the limitations of MoE architectures and show me how to implement it.”*
* *“Draft a paper summarizing the positional encoding method, and provide a step-by-step implementation guide with code for it.”*

---

## 💬 Interactive Prompting (Only in `chat` Mode)

When starting the interactive session using `python main.py chat <pdf>` (defined in [main.py](file:///e:/Vasis%20AI/main.py)), the chat interface will detect the query type and prompt you for configuration details before starting the run:

### 1. Research Paper Customization (Agent 13)
The console will prompt you to select:
* **Target Venue / Journal**:
  * `[1] IEEE` (Default)
  * `[2] NeurIPS`
  * `[3] ICML`
  * `[4] ICLR`
  * `[5] ACM`
  * `[6] Springer`
  * `[7] Elsevier`
* **Article Type**:
  * `[1] Research Article` (Default)
  * `[2] Review Article`
  * `[3] Systematic Review`
  * `[4] Short Communication`
  * `[5] Perspective Article`
  * `[6] Technical Note`
  * `[7] Case Study`
  * `[8] Letter to Editor`

### 2. Implementation Guide Customization (Agent 14)
The console will prompt you to select:
* **Your Researcher Level**:
  * `[1] Beginner`
  * `[2] Masters` (Default)
  * `[3] PhD`

---

## 💾 Automatic Output Saving

When **Agent 13 (Research Paper)** and/or **Agent 14 (Implementation Guide)** are activated, the system automatically saves their complete generated output to the [outputs/](file:///e:/Vasis%20AI/outputs) directory in the project root.

### Filename Formats
* **Agent 13 Research Papers**: 
  `paper_{venue}_{article_type}_{sanitized_topic}_{timestamp}.md`
  * *Example:* `paper_elsevier_research_article_write_a_research_paper_on_dense_20260607_234500.md`
* **Agent 14 Implementation Guides**:
  `guide_{researcher_level}_{sanitized_topic}_{timestamp}.md`
  * *Example:* `guide_masters_how_to_implement_dense_transformer_20260607_235100.md`

Each saved file starts with a clean metadata block specifying the configuration used (venue, article type, or level) and the exact generation date/time. This ensures that you don't lose any long-running generation results.
