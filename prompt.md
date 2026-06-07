# Prompt Writing Guide — PageIndex-RE-MSE 11-Agent CRDB System

**System:** PageIndex-RE-MSE (Contextual Reconstruction Database Engine)
**Engine:** 11-Agent Autonomous Pipeline (Qwen2.5-Coder 3B + DeepSeek-LLM 7B)
**Guide Version:** 2026-06-01

---

## ─── HOW THE SYSTEM READS YOUR PROMPT ───────────────────────────────────────

When you type a question, it passes through **11 agents in sequence**. The most
critical is **Agent 1 (Router)** — it classifies your prompt into one of 5
intent types. That classification determines:

- Which **penalty weights** are applied to your trust score
- Whether the query is **decomposed** into sub-questions (Agent 2)
- How **aggressively** the expansion searches for context gaps

```
Your Prompt
    │
    ▼
[Agent 1] Intent Classification  ──►  factual | comparative | definitional
    │                                 procedural | unknown
    ▼
[Agent 2] Decomposition (only for complex/multi-part prompts)
    │
    ▼
[Agent 4] Retrieval → [Agent 5] RE-MSE Expansion
    │
    ▼
[Agent 6] Grounding Validation (may trigger 1 requery if confidence < 0.60)
    │
    ▼
[Agent 7] Contradiction Audit → [Agent 9] Trust Calibration
    │
    ▼
Final Answer + TRUST LEVEL: LOW | MEDIUM | HIGH
```

---

## ─── THE 5 INTENT TYPES & THEIR CALIBRATION WEIGHTS ────────────────────────

Agent 9 applies different penalty weights based on the detected intent.
Understanding this is key to getting HIGH trust scores.

| Intent | Triggered By | Gap Weight | Gap Cap | Conflict Weight | Best For |
|---|---|---|---|---|---|
| `factual` | "What is", "What are" | 0.15 | 0.25 | 0.25 | Single-fact lookups |
| `comparative` | "How does X compare", "better than" | 0.10 | 0.20 | **0.08** | Comparing concepts |
| `definitional` | "Define", "Explain what" | **0.05** | 0.15 | 0.30 | Concept explanations |
| `procedural` | "How to", "What steps" | 0.06 | 0.15 | 0.15 | Workflows & steps |
| `causal` | "Why does", "What causes" | 0.06 | 0.15 | **0.10** | Cause-and-effect reasoning |
| `unknown` | Ambiguous/mixed | 0.15 | 0.25 | 0.20 | Avoid this |

> **Key insight:** `definitional` has the lowest gap penalty (0.05). If your question
> is about explaining a concept, phrase it as definitional to get the best trust score.

---

## ─── INTENT TYPE GUIDE & EXAMPLE PROMPTS ───────────────────────────────────

### 🔵 FACTUAL — Single precise facts

Triggers when your question asks for a specific value, number, name, or
attribute. Best for pinpoint lookups.

**Optimal phrasing:**
```
What is the learning rate used in training?
What is the dimension of the key vectors (d_k) in Scaled Dot-Product Attention?
What dataset was used to evaluate this model?
How many parameters does this architecture have?
```

**Trust expectation:** MEDIUM–HIGH (if fact exists in document)
**Calibration:** gap_weight=0.15, conflict_weight=0.25 → avoid papers where the
same fact appears with different values on different pages.

---

### 🟡 COMPARATIVE — Evaluating differences or superiority

Triggers when your question asks to contrast, compare, or evaluate one thing
against another. Has the **lowest conflict penalty (0.08)** — ideal for topics
where the document presents multiple perspectives.

**Optimal phrasing:**
```
How does max pooling compare to average pooling in CNNs?
Why are CNNs better than traditional neural networks for image recognition?
What are the advantages of self-attention over recurrent neural networks?
How does the Transformer's encoder differ from its decoder?
```

**Trust expectation:** MEDIUM–HIGH
**Calibration:** gap_weight=0.10, conflict_weight=0.08 → very forgiving of
conflicting viewpoints (by design — comparisons inherently discuss trade-offs).

---

### 🟢 DEFINITIONAL — Explaining what something is

Triggers when asking for a definition, explanation, or conceptual description.
Has the **lowest gap penalty (0.05)** — best intent for in-depth concept questions
since the answer usually lives in a few focused pages.

**Optimal phrasing:**
```
What is a convolutional layer?
Explain the role of the ReLU activation function in neural networks.
Define the attention mechanism in Transformers.
What is backpropagation and how does it work?
```

**Trust expectation:** HIGH (definitions are dense and well-localized)
**Calibration:** gap_weight=0.05, conflict_weight=0.30 → gaps barely hurt, but
avoid asking about things defined differently in multiple sections.

---

### 🟠 PROCEDURAL — Steps, workflows, processes

Triggers when asking how to do something, or what the sequence of steps is.
Steps span many pages naturally — gap penalty is kept low (0.06).

**Optimal phrasing:**
```
What are the steps involved in training a CNN from scratch?
How is backpropagation performed in a neural network?
Describe the process of forward propagation in a Transformer.
What is the training procedure used in this paper?
```

**Trust expectation:** MEDIUM (steps span many pages, gaps are expected)
**Calibration:** gap_weight=0.06, conflict_weight=0.15.

---

### 🟣 CAUSAL — Why something happens or what causes an effect

NEW intent type — triggers on "Why", "What causes", "What leads to", "How does
X improve/affect Y". Has the **lowest conflict weight (0.10)** — causal chains
naturally have multiple valid explanations so soft inconsistencies are expected.

**Optimal phrasing:**
```
Why does increasing the depth of a CNN improve feature extraction?
What causes the vanishing gradient problem in deep networks?
What leads to overfitting and how does dropout address it?
How does adding more convolutional layers affect accuracy?
Why is the ReLU function preferred over sigmoid in deep networks?
```

**Trust expectation:** MEDIUM–HIGH
**Calibration:** gap_weight=0.06, conflict_weight=0.10, gap_cap=0.15 — very
forgiving. Causal reasoning spans many pages and involves nuanced trade-offs.

---

### 🔴 HALLUCINATION REJECTION — Testing out-of-scope topics

A powerful negative control. Ask about something you know is **not** in the
document. The system should return LOW trust + rejection message.

**Use these to verify the system is not hallucinating:**
```
What quantum computing experiments are mentioned in this paper?
What blockchain technology is described in this document?
What is the revenue of the company mentioned in this paper?
```

**Expected result:** `TRUST LEVEL: LOW` + answer containing "not mentioned",
"not discussed", or "does not contain".

---

## ─── COMPLEX / MULTI-PART PROMPTS ──────────────────────────────────────────

Agent 2 (Decomposer) activates when your prompt is marked `is_complex=true`
by Agent 1. It splits into up to `MAX_SUB_QUERIES` independent sub-questions.

**When to use complex prompts:**
- You want to cover multiple concepts in one query
- You want a comprehensive survey of a topic

**How to write them:**
```
# Good complex prompt (2-3 clear distinct questions joined):
What is a convolutional layer, how does it differ from a fully connected layer,
and what role does dropout play in preventing overfitting?

# Another good one:
Explain the encoder and decoder components of the Transformer and describe how
masking is applied in the decoder's self-attention.
```

**⚠ Caution with complex prompts:**
- Each sub-query runs its own retrieval → longer runtime
- More sub-queries = more context gaps detected = slightly lower trust
- Keep to 2–3 distinct parts maximum for best results

---

## ─── TRUST SCORE MECHANICS ─────────────────────────────────────────────────

Your final trust score is calculated as:

```
trust_score = base_score
            - gap_penalty        (context holes in the reconstructed answer)
            - conflict_penalty   (factual contradictions found)
            + temporal_bonus     (if document has time-ordered structure, +0.05)
            - requery_penalty    (if grounding failed once, -0.05, max -0.10)
```

**Trust levels:**
| Score | Level | Meaning |
|---|---|---|
| ≥ 0.75 | `HIGH` | Strongly grounded, no conflicts |
| ≥ 0.60 | `MEDIUM` | Grounded with minor gaps or soft inconsistencies |
| < 0.60 | `LOW` | Weak grounding, out-of-scope, or strong conflicts |

**How to maximize trust score:**

1. **Ask focused, single-topic questions** — fewer context gaps
2. **Use `definitional` or `comparative` phrasing** — lowest gap penalties
3. **Ask about things well-covered in the document** — high base score from Agent 6
4. **Avoid mixing unrelated concepts** in one prompt

---

## ─── WRITING DO'S AND DON'TS ───────────────────────────────────────────────

### ✅ DO

| Pattern | Example | Why It Works |
|---|---|---|
| Start with "What is" | "What is the role of pooling layers?" | → `definitional`, gap_weight=0.05 |
| Use "compare" / "differ" | "How does X differ from Y?" | → `comparative`, conflict_weight=0.08 |
| Use "steps" / "process" | "What steps are involved in...?" | → `procedural`, expected multi-page |
| Be specific with entities | "the ReLU activation function" | Helps Agent 4 find the right atoms |
| Ask one thing at a time | Single-concept questions | Fewer gaps, higher trust |

### ❌ DON'T

| Pattern | Problem | Better Alternative |
|---|---|---|
| Vague queries | "Tell me everything about CNNs" | "What is the architecture of a CNN?" |
| Mix 4+ concepts | "Explain ReLU, pooling, dropout, and batch norm" | Split into 2 separate queries |
| Ask what's not in the paper | "What GPU was used?" (if not mentioned) | Use as negative control test instead |
| ALL CAPS + typos | "WHY RNN IS BETTER THAN OTHER" | "Why are RNNs better suited than other architectures?" |
| Add "is this correct?" | Ambiguous phrasing | Ask a direct question |

---

## ─── INTENT TRIGGER KEYWORDS (QUICK REFERENCE) ─────────────────────────────

Use these leading words to steer Agent 1 toward the right intent:

```
FACTUAL      → "What is", "What are", "How many", "Which", "When was",
               "What value", "What number"

COMPARATIVE  → "Compare", "How does X differ from Y", "Why is X better than Y",
               "Advantages of X over Y", "X vs Y", "What makes X different"

DEFINITIONAL → "Define", "What is meant by", "Explain what", "Describe the concept",
               "What does X mean", "What is X in the context of"

PROCEDURAL   → "How to", "What are the steps", "Describe the process",
               "How is X performed", "What is the procedure", "Walk me through"

CAUSAL       → "Why does", "What causes", "What leads to", "How does X improve",
               "How does X affect", "Why is X better", "What makes X work",
               "Why does X happen", "What is the reason for"

AVOID        → Starting with just "Tell me", "Can you", "Is this" — these
               often map to `unknown` intent (worst penalty weights)
```

---

## ─── EXAMPLE PROMPT SETS BY DOCUMENT TYPE ──────────────────────────────────

### For a Neural Network / Deep Learning Paper

```
# Definitional (best trust)
What is the attention mechanism and how does it work?
Define the role of the encoder in a Transformer architecture.

# Comparative (lowest conflict penalty)
How does self-attention differ from recurrent neural networks?
Compare max pooling and average pooling in terms of spatial invariance.

# Factual (precise lookups)
What is the value of d_k used in Scaled Dot-Product Attention?
How many attention heads are used in the multi-head attention layer?

# Procedural
What are the steps for training a Transformer model from scratch?
How is the forward pass computed through a convolutional layer?

# Causal (multi-hop reasoning)
Why does increasing depth in CNNs improve feature extraction?
What causes the vanishing gradient problem and how does ReLU address it?

# Negative Control (hallucination test)
What quantum hardware accelerators are used in this paper?
```

### For a Business / Research Report

```
# Definitional
What is the main research objective described in this paper?

# Factual
What accuracy metric was reported on the test dataset?

# Comparative
How does the proposed method compare to the baseline?

# Procedural
What experimental methodology was followed in this study?
```

---

## ─── READING THE OUTPUT ─────────────────────────────────────────────────────

After every query, read these key output signals:

```
[Agent1] intent=comparative    ← Check this matches your intended type
[Agent5] 14 atoms, 6 pages     ← More pages = more gaps expected
[Agent6] confidence=0.85       ← Base score for calibration
[Agent9] calibrated_score=0.70 ← Final trust score
TRUST LEVEL: MEDIUM (0.70)     ← Your reliability indicator
```

**If you see LOW trust on a grounded answer:**
- Check Agent1 intent — did it classify correctly?
- Check Agent5 — too many pages/gaps for your query?
- Try rephrasing as `definitional` if possible (lowest gap weight)

**If you see `[WARNING] CONTRADICTIONS DETECTED!`:**
- This is a **structural** conflict in the knowledge graph (real)
- The document likely defines the same fact differently in multiple places
- Cross-reference the provenance sections listed to find the conflict

---

*Last updated: 2026-06-01 | System: PageIndex-RE-MSE v1 | Engine: 11-Agent CRDB*
