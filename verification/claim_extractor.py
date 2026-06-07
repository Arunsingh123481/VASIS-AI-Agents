"""
Claim Entity Extractor — Step 1 of the Verification Retrieval Pipeline.

Extracts:
  - main_entity   : primary subject of the claim (e.g. "Transformer")
  - required_terms: secondary concepts that must appear in evidence
  - relation      : the verbal relation (e.g. "removes", "relies on")
  - target        : what the relation acts on
  - all_keywords  : flat deduplicated list for BM25/keyword search
"""

import re
from typing import Dict, List


# ---------------------------------------------------------------------------
# Curated term taxonomies per research domain
# ---------------------------------------------------------------------------

_DOMAIN_TERMS: Dict[str, List[str]] = {
    "transformer": [
        "transformer", "attention", "self-attention", "multi-head",
        "encoder", "decoder", "recurrence", "recurrent", "sequential",
        "positional encoding", "feed-forward", "attention mechanism",
        "attention is all you need", "vaswani",
    ],
    "cnn": [
        "convolutional", "convolution", "pooling", "feature map",
        "spatial", "image", "kernel", "filter", "receptive field",
        "inception", "resnet", "vgg", "alexnet",
    ],
    "rnn": [
        "recurrent", "lstm", "gru", "hidden state", "sequence",
        "time step", "vanishing gradient", "backpropagation through time",
    ],
    "bert": [
        "bert", "bidirectional", "masked language model", "pre-training",
        "fine-tuning", "wordpiece", "next sentence prediction",
    ],
    "diffusion": [
        "diffusion", "denoising", "noise schedule", "score matching",
        "ddpm", "latent diffusion", "stable diffusion",
    ],
    "gan": [
        "generative adversarial", "discriminator", "generator",
        "adversarial training", "mode collapse",
    ],
    "reinforcement": [
        "reward", "policy", "agent", "q-learning", "actor-critic",
        "markov decision process", "value function",
    ],
    "overfitting": [
        "overfitting", "regularization", "dropout", "weight decay",
        "generalization", "bias-variance",
    ],
}

# Semantic relation groups
_RELATION_GROUPS = [
    {
        "canonical": "remove",
        "equivalents": [
            "dispense with", "eliminate", "avoid", "replace", "eschew", "remove", "discard"
        ]
    },
    {
        "canonical": "rely on",
        "equivalents": [
            "based on", "uses", "depends on", "relies on", "built upon", "employ", "leverage"
        ]
    },
    {
        "canonical": "improve",
        "equivalents": [
            "enhance", "increase", "boost", "optimize", "improve", "accelerate"
        ]
    },
    {
        "canonical": "reduce",
        "equivalents": [
            "decrease", "minimize", "lower", "reduce", "mitigate"
        ]
    },
    {
        "canonical": "introduce",
        "equivalents": [
            "propose", "present", "contribute", "introduce"
        ]
    },
    {
        "canonical": "outperform",
        "equivalents": [
            "surpass", "beat", "exceed", "outperform"
        ]
    },
    {
        "canonical": "achieve",
        "equivalents": [
            "reach", "obtain", "demonstrate", "achieve"
        ]
    },
    {
        "canonical": "prevent",
        "equivalents": [
            "prevent", "avoid"
        ]
    },
    {
        "canonical": "consist of",
        "equivalents": [
            "is based on", "is composed of", "consists of", "comprises"
        ]
    },
    {
        "canonical": "enable",
        "equivalents": [
            "allow", "permit", "facilitate", "enable"
        ]
    }
]

# Semantic Opposition Groups
_OPPOSITION_MAP = [
    {
        "concept": "recurrent",
        "opposites": [
            "recurrence-free", "without recurrence", "dispensing with recurrence", 
            "eschewing recurrence", "replaces recurrent layers", "no recurrent units"
        ]
    },
    {
        "concept": "lstm",
        "opposites": [
            "transformer", "attention-only", "self-attention"
        ]
    },
    {
        "concept": "cnn",
        "opposites": [
            "attention-only", "transformer"
        ]
    },
    {
        "concept": "transformer",
        "opposites": [
            "lstm-based", "rnn-based", "convolutional", "recurrence-based"
        ]
    }
]

# Exclusion / Negative phrases
_EXCLUSION_TERMS = [
    "replacing", "instead of", "dispensing with", "eschewing", "no recurrent",
    "without using", "not based on", "avoids", "removes", "eliminates", "only", "solely"
]

def extract_claim_entities(claim: str) -> Dict:
    """
    Parse a claim sentence into structured entities for retrieval targeting.

    Returns:
        {
          "main_entity":    str | None,
          "required_terms": list[str],
          "relation":       str | None,
          "canonical_relation": str | None,
          "target":         str | None,
          "domain":         str | None,
          "all_keywords":   list[str],
          "architecture":   str | None,
          "paradigm":       str | None,
          "exclusions":     list[str],
          "opposition_risks": list[str]
        }
    """
    if not isinstance(claim, str):
        claim = ""
    claim_lower = claim.lower()
    result = {
        "main_entity": None,
        "required_terms": [],
        "relation": None,
        "canonical_relation": None,
        "target": None,
        "domain": None,
        "all_keywords": [],
        "architecture": None,
        "paradigm": None,
        "exclusions": [],
        "opposition_risks": []
    }

    # ── 1. Detect domain & collect required terms ──────────────────────────
    matched_domain = None
    matched_terms = []
    for domain, terms in _DOMAIN_TERMS.items():
        hits = [t for t in terms if t in claim_lower]
        if hits and len(hits) > len(matched_terms):
            matched_domain = domain
            matched_terms = hits

    result["domain"] = matched_domain
    result["required_terms"] = matched_terms

    # ── 2. Infer main entity, architecture, paradigm ──────────────────────
    _domain_entity_map = {
        "transformer": "Transformer",
        "cnn": "CNN",
        "rnn": "RNN",
        "bert": "BERT",
        "diffusion": "Diffusion Model",
        "gan": "GAN",
        "reinforcement": "Reinforcement Learning",
        "overfitting": "neural network",
    }
    if matched_domain:
        result["main_entity"] = _domain_entity_map.get(matched_domain)
        result["architecture"] = result["main_entity"]
        
    # Heuristics for Paradigm
    if "recurrent" in claim_lower or "lstm" in claim_lower or "rnn" in claim_lower:
        result["paradigm"] = "recurrent"
    elif "attention" in claim_lower:
        result["paradigm"] = "attention"
    elif "convolution" in claim_lower or "cnn" in claim_lower:
        result["paradigm"] = "convolutional"

    # Specific Architecture overrides for exact text
    if "lstm-based" in claim_lower:
        result["architecture"] = "LSTM"
    elif "transformer architecture" in claim_lower:
        result["architecture"] = "Transformer"

    # Fallback: extract first capitalised noun phrase up to 4 words
    if not result["main_entity"]:
        cap_match = re.match(r'^([A-Z][a-zA-Z\-]+(?: [A-Z][a-zA-Z\-]+){0,3})', claim.strip())
        if cap_match:
            result["main_entity"] = cap_match.group(1)

    # ── 3. Extract relation verb and canonicalize ─────────────────────────
    best_match_idx = -1
    for group in _RELATION_GROUPS:
        for equiv in group["equivalents"]:
            # Simple boundary match
            pat = r'\b' + re.escape(equiv) + r'(s|es|d|ed|ing)?\b'
            m = re.search(pat, claim_lower)
            if m:
                idx = m.start()
                if best_match_idx == -1 or idx < best_match_idx:
                    best_match_idx = idx
                    # Original text from claim
                    original_rel = claim[m.start():m.end()]
                    result["relation"] = original_rel
                    result["canonical_relation"] = group["canonical"]

    # ── 4. Extract target (phrase after relation) ─────────────────────────
    if result["relation"]:
        idx = claim_lower.find(result["relation"].lower())
        after = claim[idx + len(result["relation"]):].strip()
        # Take up to 8 words, strip trailing punctuation
        words = after.split()[:8]
        result["target"] = " ".join(words).rstrip(".,;:") or None

    # ── 5. Detect exclusions and opposition risks ──────────────────────────
    for ex in _EXCLUSION_TERMS:
        if ex in claim_lower:
            result["exclusions"].append(ex)
            
    for opp in _OPPOSITION_MAP:
        if opp["concept"] in claim_lower:
            result["opposition_risks"].extend(opp["opposites"])
        for phrase in opp["opposites"]:
            if phrase in claim_lower:
                result["opposition_risks"].append(opp["concept"])

    # deduplicate
    result["opposition_risks"] = list(set(result["opposition_risks"]))

    # ── 6. Build flat keyword list (for BM25 / entity overlap) ───────────
    kws = set()
    if result["main_entity"]:
        kws.add(result["main_entity"].lower())
    for t in result["required_terms"]:
        kws.add(t.lower())
    if result["canonical_relation"]:
        kws.add(result["canonical_relation"].lower())
    if result["target"]:
        # Add significant words from target
        stop = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "with", "by", "its", "only"}
        for w in result["target"].lower().split():
            w_clean = re.sub(r'[^a-z0-9]', '', w)
            if w_clean and w_clean not in stop and len(w_clean) > 2:
                kws.add(w_clean)

    result["all_keywords"] = sorted(kws)
    return result
