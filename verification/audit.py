"""
DraftVerifier — Verification Retrieval Pipeline (Upgraded).

Full pipeline for audit_sentence():

  Claim
  ↓ Step 1: Claim entity extraction       (claim_extractor)
  ↓ Step 2: Paper-level filtering         (paper_filter)
  ↓ Step 3: Section-level filtering       (PageIndex tree navigation)
  ↓ Step 4: Hybrid retrieval              (hybrid_retriever — BM25 + vector + entity)
  ↓ Step 5: Reranking                     (reranker — BGE-Reranker-v2-m3 / keyword fallback)
  ↓ Step 6: Evidence validation           (evidence_validator — entity gate)
  ↓ Step 7: Insufficient-evidence check   → "No reliable evidence found" fallback
  ↓ Step 8: LLM-based status judgment     (ask_llm with tight prompt)
  ↓ Final status
"""

import sys
import os
import json
from typing import Dict, Any, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from llm.ollama_client import ask_llm

from verification.claim_extractor  import extract_claim_entities
from verification.paper_filter     import filter_papers
from verification.hybrid_retriever import hybrid_retrieve
from verification.reranker         import rerank
from verification.evidence_validator import validate_evidence


# ── Constants ─────────────────────────────────────────────────────────────────

TOP_K_HYBRID    = 20   # candidates fed to the reranker
TOP_K_EVIDENCE  = 5    # final evidence atoms shown to LLM / frontend
MIN_FINAL_SCORE = 0.10 # minimum composite score to pass evidence gate


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_insufficient_response(sentence: str) -> Dict[str, Any]:
    """Return a clean 'no evidence found' result (never force bad evidence)."""
    return {
        "status": "Unsupported",
        "score": 5,
        "explanation": (
            "No reliable supporting evidence found in the uploaded papers. "
            "The retrieval pipeline could not locate atoms that match the claim's "
            "main entity and required terms. Consider adding the relevant source paper."
        ),
        "closest_match": "",
        "correction": "",
        "pages": [],
        "sections": [],
        "anchor_atoms": [],
        "atom_range": "",
        "evidence": [],
        "claim_coverage": [],
        "pipeline_debug": {
            "claim_entities": {},
            "papers_searched": 0,
            "candidates_before_validation": 0,
            "candidates_after_validation": 0,
            "insufficient_evidence": True,
        }
    }


def _build_evidence_list(atoms: List[Dict]) -> List[Dict]:
    return [
        {
            "text":             a.get("text", ""),
            "source":           a.get("source_doc", a.get("doc_id", "Unknown")),
            "page":             a.get("page_num"),
            "score":            round(a.get("_final_score", a.get("_prescore", 0.0)), 4),
            "section":          a.get("section", ""),
            "paragraph_index":  a.get("paragraph_index", a.get("local_idx")),
            "atom_id":          a.get("atom_id"),
        }
        for a in atoms
    ]


# ── Main verifier class ───────────────────────────────────────────────────────

class DraftVerifier:
    def __init__(self):
        pass

    # ── Autocomplete (unchanged) ──────────────────────────────────────────────
    def autocomplete(self, rag_instance, text_context: str) -> Dict[str, Any]:
        """Vault-grounded autocomplete: takes the last 200 chars and completes 1-2 sentences."""
        query = f"Continue this thought related to the research paper: {text_context}"
        result = rag_instance.query(query, show_provenance=False, save_result=False)
        narrative = result.get("narrative", "")
        ordered_atoms = result.get("ordered_atoms", [])

        prompt = f"""You are a helpful academic writing assistant.
The student is writing a paper and needs help completing their thought.
Based ONLY on the provided context, write the next 1-2 sentences to seamlessly continue the student's text.

Student's text: "{text_context}"

Context from vault:
---
{narrative}
---

Return ONLY the continuation sentences. Do not repeat the student's text. If the context does not support a completion, reply with an empty string.
"""
        completion = ask_llm(prompt)
        return {
            "completion": completion,
            "source_atoms": [a.get("atom_id") for a in ordered_atoms if "atom_id" in a]
        }

    # ── Core verification pipeline ────────────────────────────────────────────
    def audit_sentence(
        self,
        rag_instance,
        sentence: str,
        all_sessions: Optional[Dict] = None,   # { session_id: rag } for multi-paper search
    ) -> Dict[str, Any]:
        """
        Full 8-step verification retrieval pipeline.

        Parameters
        ----------
        rag_instance : the primary PageIndexREMSE instance for this session
        sentence     : the claim to verify
        all_sessions : optional dict of all sessions in the same vault
                       (enables cross-paper search)
        """

        # ── Step 1: Extract claim entities ────────────────────────────────
        claim_entities = extract_claim_entities(sentence)
        print(f"[audit] Claim entities: {claim_entities}")

        # ── Step 2: Paper-level filtering ────────────────────────────────
        sessions_to_search: Dict = {}
        if all_sessions and len(all_sessions) > 1:
            ranked_sessions = filter_papers(all_sessions, claim_entities, top_n=3)
            for sid, rag, score in ranked_sessions:
                sessions_to_search[sid] = rag
            print(f"[audit] Paper filter selected {len(sessions_to_search)} sessions "
                  f"(from {len(all_sessions)} total)")
        else:
            # single-session mode
            sessions_to_search = {"_primary": rag_instance}

        # ── Step 3: Section-level filtering via PageIndex tree ───────────
        # Collect atoms only from structurally relevant sections
        all_atoms: List[Dict] = []
        for sid, rag in sessions_to_search.items():
            try:
                from retrieval.tree_navigator import navigate_tree
                selected_sections, _ = navigate_tree(sentence, rag.tree_nodes)
                valid_pages: set = set()
                for s in selected_sections:
                    valid_pages.update(range(s["start_page"], s["end_page"] + 1))

                # Prefer section-scoped atoms; fall back to all atoms if too few
                section_atoms = [
                    a for a in rag.atoms if a.get("page_num") in valid_pages
                ]
                if len(section_atoms) < 10:
                    section_atoms = rag.atoms   # fallback: use all
                all_atoms.extend(section_atoms)
                print(f"[audit] Session {sid}: {len(section_atoms)} section-scoped atoms")
            except Exception as e:
                print(f"[audit] Tree navigation failed for {sid}: {e} — using all atoms")
                all_atoms.extend(getattr(rag, "atoms", []))

        if not all_atoms:
            return _make_insufficient_response(sentence)

        # ── Step 4: Hybrid retrieval (BM25 + vector + entity) ────────────
        candidates = hybrid_retrieve(
            claim=sentence,
            atoms=all_atoms,
            claim_entities=claim_entities,
            top_k=TOP_K_HYBRID,
        )
        candidates_before = len(candidates)
        print(f"[audit] Hybrid retrieval: {candidates_before} candidates")

        # ── Step 5: Reranking ────────────────────────────────────────────
        reranked = rerank(claim=sentence, candidates=candidates)

        # ── Step 6: Evidence validation (entity gate) ────────────────────
        valid_atoms, rejected_atoms = validate_evidence(
            atoms=reranked,
            claim_entities=claim_entities,
            min_final_score=MIN_FINAL_SCORE,
        )
        print(f"[audit] Validation: {len(valid_atoms)} passed, {len(rejected_atoms)} rejected")
        if rejected_atoms:
            for r in rejected_atoms[:3]:
                print(f"  ✗ rejected ({r.get('_rejection_reason', '?')}): "
                      f"{r.get('text', '')[:60]}…")

        # ── Step 7: Insufficient-evidence fallback ────────────────────────
        if not valid_atoms:
            print("[audit] No valid atoms passed entity gate → insufficient evidence")
            result = _make_insufficient_response(sentence)
            result["pipeline_debug"]["claim_entities"] = claim_entities
            result["pipeline_debug"]["papers_searched"] = len(sessions_to_search)
            result["pipeline_debug"]["candidates_before_validation"] = candidates_before
            return result

        # Take the best evidence atoms for the LLM
        top_atoms = valid_atoms[:TOP_K_EVIDENCE]
        evidence_texts = "\n".join(
            f"[Atom {a.get('atom_id', '?')} | Page {a.get('page_num', '?')} | "
            f"Score {a.get('_final_score', 0):.3f}]\n{a.get('text', '')}"
            for a in top_atoms
        )

        # ── Step 8: LLM-based status judgment ────────────────────────────
        pages   = list({a.get("page_num") for a in top_atoms if a.get("page_num") is not None})
        sections = list({a.get("section", "") for a in top_atoms if a.get("section")})
        atom_ids = [a.get("atom_id") for a in top_atoms if a.get("atom_id") is not None]
        atom_range = (
            f"{atom_ids[0]} → {atom_ids[-1]}" if len(atom_ids) > 1 else str(atom_ids[0])
        ) if atom_ids else ""

        canonical_rel_text = (
            f"\nCanonical Relation Mapping: The claim's relation '{claim_entities.get('relation')}' is semantically "
            f"normalized to '{claim_entities.get('canonical_relation')}'. Treat synonyms (e.g., dispense with, eschew, eliminate) "
            f"as semantically equivalent to this canonical relation."
        ) if claim_entities.get("canonical_relation") else ""

        opp_risks = ", ".join(claim_entities.get("opposition_risks", []))
        opp_text = f"\nOpposition Risks: Look out for these opposing concepts in the evidence: {opp_risks}. If they appear, it is a Contradiction." if opp_risks else ""
        
        arch = claim_entities.get("architecture")
        arch_text = f"\nClaimed Architecture: {arch}" if arch else ""
        
        paradigm = claim_entities.get("paradigm")
        paradigm_text = f"\nClaimed Paradigm: {paradigm}" if paradigm else ""

        prompt = f"""You are an advanced academic verification engine performing Contradiction-Aware Semantic Scoring.
Compare the student's claim ONLY against the retrieved evidence below. You must actively look for BOTH support and semantic opposition/contradiction.

Student's Claim:
"{sentence}"{canonical_rel_text}{opp_text}{arch_text}{paradigm_text}

Retrieved Evidence (entity-validated):
---
{evidence_texts}
---

Determine the verification status by evaluating support vs contradiction:
1. "Verified"             — Claim exactly matches the evidence (support > 90, contradiction = 0).
2. "Semantically Supported" — Meaning matches but wording differs (support > 75, contradiction = 0).
3. "Partially Supported"  — Some parts match but others are unsupported.
4. "Contradicted"         — Evidence explicitly disagrees with the claim (contradiction_score > support_score). Look for architecture conflicts (e.g., LSTM vs Transformer) or negation/replacement patterns (e.g., "dispenses with recurrence", "replaces", "instead of").
5. "Unsupported"          — Evidence does not support the claim at all.

IMPORTANT RULES FOR CONTRADICTION:
- If the claim says "LSTM-based" and evidence says "Transformer" or "dispenses with recurrence", this is CONTRADICTED.
- Do not mistake semantic opposition (e.g., "dispensing with X") for support of "uses X".
- Explain the contradiction clearly in the 'explanation' field.

Return ONLY a valid JSON object:
{{
    "status": "Verified" | "Semantically Supported" | "Partially Supported" | "Contradicted" | "Unsupported",
    "score": <0-100 integer support score>,
    "contradiction_score": <0-100 integer contradiction score>,
    "explanation": "<Nuanced reasoning. If contradicted: 'The claim is contradicted because evidence explicitly states X, which conflicts with the claim's architecture Y.'>",
    "closest_match": "<The most relevant exact sentence from the evidence>",
    "correction": "<Suggested safer correction if contradicted/partial/unsupported, else empty>",
    "pages": {json.dumps(pages)},
    "sections": {json.dumps(sections)},
    "claim_coverage": [
        {{"part": "<key phrase from claim>", "status": "supported"|"semantically supported"|"partially supported"|"unsupported"|"contradicted", "explanation": "<why this status was given>"}}
    ]
}}"""

        response = ask_llm(prompt, expect_json=True)
        try:
            audit = json.loads(response)
        except Exception as e:
            print(f"[audit] JSON parse error: {e} — raw: {response[:200]}")
            audit = {
                "status": "Needs Review",
                "score": 30,
                "explanation": "Audit ran successfully but LLM returned malformed JSON. Manual review recommended.",
                "closest_match": top_atoms[0].get("text", "")[:200] if top_atoms else "",
                "correction": "",
                "pages": pages,
                "sections": sections,
                "claim_coverage": [],
            }

        # Attach retrieval metadata
        audit["anchor_atoms"]  = atom_ids
        audit["atom_range"]    = atom_range
        audit["evidence"]      = _build_evidence_list(top_atoms)
        audit["pipeline_debug"] = {
            "claim_entities":               claim_entities,
            "papers_searched":              len(sessions_to_search),
            "candidates_before_validation": candidates_before,
            "candidates_after_validation":  len(valid_atoms),
            "rejected_count":               len(rejected_atoms),
            "insufficient_evidence":        False,
        }

        return audit


verifier_engine = DraftVerifier()
