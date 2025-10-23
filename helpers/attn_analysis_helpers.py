# utils_logic_attn.py
from typing import List, Dict, Tuple, Optional, Any
import torch

Span = Tuple[int, int]  # [start, end) token indices (0-based, end exclusive)

# -------------------------------
# Token-ID region identification helpers
# -------------------------------

# Tokenize text to a flat python list of token IDs, no BOS since not relevant
def _to_ids(model, text: str) -> List[int]:
    return model.to_tokens(text, prepend_bos=False)[0].tolist()

# Find token IDs for searching a clause (handles spaces/newlines & trailing punctuation).
# Returns a list of token-id lists.
def _make_variants(model, text: str) -> List[List[int]]:
    leads = ["", " ", "\n"]
    trails = ["", ".", ";", " .", " ;"]
    bag = []
    for L in leads:
        for T in trails:
            bag.append(L + text + T)
    # Also allow these common endings
    bag += [text + ". ", text + "; "]

    # De-duplicate by token sequence
    seen = set()
    out: List[List[int]] = []
    for s in bag:
        ids = _to_ids(model, s)
        key = tuple(ids)
        if len(ids) > 0 and key not in seen:
            seen.add(key)
            out.append(ids)
    return out

# Find a subsequence 'needle' of token IDs in the full tokenized prompt 'hay'
def _find_all_subseq(sntc: List[int], subseq: List[int]) -> List[int]:
    out = []
    if len(subseq) == 0 or len(subseq) > len(sntc):
        return out
    n = len(subseq)
    for i in range(len(sntc) - n + 1):
        if sntc[i:i+n] == subseq:
            out.append(i)
    return out

def _find_marker(tokens_row: List[int], 
                 model, marker: str,
                 start_at: int = 0, 
                 end_at: Optional[int] = None,
                 want_last: bool = False) -> Optional[Span]:

    """ 
    Find markers for the logic problem, including 'Rules:',  'Facts:', 'Question:', 'Answer:'.
    Returns (start, end) token indices or None.
    """
    if end_at is None:
        end_at = len(tokens_row)
    hay = tokens_row[start_at:end_at]

    variants = []
    for s in [marker, " " + marker, "\n" + marker]:
        variants.append(_to_ids(model, s))

    best: Optional[Tuple[int,int]] = None
    for v in variants:
        starts = _find_all_subseq(hay, v)
        if not starts:
            continue
        idx = (starts[-1] if want_last else starts[0]) + start_at
        cand = (idx, idx + len(v))
        if best is None:
            best = cand
        else:
            if want_last and cand[0] > best[0]:
                best = cand
            if (not want_last) and cand[0] < best[0]:
                best = cand
    return best


def _find_in_range(tokens_row: List[int], model, clause_text: str,
                   start_: int, end_: int) -> Optional[Span]:
    """
    Find clause_text (as token-ID sequence) within [start_bound, end_bound) using region markers.
    For our purposes, the clause_text would fall into the categories of queried rule and correct fact,
        since this is primarily used for finding the attention mass of a selected attention head on such clauses.
    
    Returns (start, end) or None.
    """
    region = tokens_row[start_:end_]
    variants = _make_variants(model, clause_text)

    for v in variants:
        starts = _find_all_subseq(region, v)
        if starts:
            s = starts[0] + start_
            return (s, s + len(v))

    # If direct match fails, try seeking a trailing punctuation token in the region.
    base_variants = _make_variants(model, clause_text.rstrip(".; "))
    for v in base_variants:
        for punct in [".", ";"]:
            v2 = v + _to_ids(model, punct)
            starts = _find_all_subseq(hay, v2)
            if starts:
                s = starts[0] + start_bound
                return (s, s + len(v2))
    return None

# Finding different regions of the last problem (after the in-context examples).
# Useful for cleanly locating the queried rule and correct fact for writing the proof.
def locate_final_problem_regions(tokens_row: torch.Tensor, model) -> Dict[str, Span]:
    """
    Returns token-ID level bounds (start, end) for the last problem in the prompt:
    - rules_region: from end of 'Rules:' to start of 'Facts:'
    - facts_region: from end of 'Facts:' to start of 'Question:'
    - problem_region: from 'Rules:' to 'Answer:' for the last problem (inclusive on both ends)
    - answer_marker: the 'Answer:' marker span
    """
    row = tokens_row.tolist()

    # Rules region (the LogOp chain and linear chain)
    rules_m = _find_marker(row, model, "Rules:", want_last=True)
    rules_start, rules_end = rules_m

    # Facts region (the truth-value assignment sentences)
    facts_m = _find_marker(row, model, "Facts:", start_at=rules_end, want_last=False)
    facts_start, facts_end = facts_m

    # Question region ("Question: state the truth value of QUERY.")
    question_m = _find_marker(row, model, "Question:", start_at=facts_end, want_last=False)
    answer_m   = _find_marker(row, model, "Answer:",   start_at=facts_end, want_last=False)
    q_start, q_end = (answer_m if question_m is None else question_m)

    final_answer_m = _find_marker(row, model, "Answer:", start_at=q_end, want_last=False)
    if final_answer_m is None:
        if answer_m is not None and answer_m[0] >= facts_end:
            final_answer_m = answer_m
        else:
            raise ValueError("Could not find final 'Answer:' for the last problem.")
    ans_start, ans_end = final_answer_m

    return {
        "rules_region": (rules_end, facts_start),
        "facts_region": (facts_end, q_start),
        "problem_region": (rules_start, ans_end),
        "answer_marker": (ans_start, ans_end),
    }

# Identifying the position of a clause (e.g. a rule, a fact) in the
# final problem (after the in-context examples), for a single sample.
def clause_token_spans_for_row(tokens_row: torch.Tensor,
                               model,
                               problem_info: Dict[str, str]) -> Dict[str, Optional[Span]]:
    """
    Given a single prompt (token ID sequence) and a dict like
      {'queried_rule': 'B implies E', 'correct_fact': 'B is true'}
    return token spans (start, end) for each clause inside the final problem.
    """
    regions = locate_final_problem_regions(tokens_row, model)
    row = tokens_row.tolist()

    out: Dict[str, Optional[Span]] = {"queried_rule": None, "correct_fact": None}

    # Queried rule lives in the Rules region
    if problem_info.get("queried_rule"):
        out["queried_rule"] = _find_in_range(row, model, problem_info["queried_rule"],
                                             *regions["rules_region"])

    # Correct fact lives in the Facts region
    if problem_info.get("correct_fact"):
        out["correct_fact"] = _find_in_range(row, model, problem_info["correct_fact"],
                                             *regions["facts_region"])

    return out

# Invoking clause_token_spans_for_row for a batch of samples.
def clause_token_spans_for_batch(clean_tokens: torch.Tensor,
                                 model,
                                 problem_infos: List[Dict[str, str]]) -> List[Dict[str, Optional[Span]]]:
    """
    Batch version of clause_token_spans_for_row.
    clean_tokens is [B, S]; problem_infos length must be B.
    Returns a list of dicts with 'queried_rule' and 'correct_fact' spans per row.
    """
    B = clean_tokens.shape[0]
    assert len(problem_infos) == B, "problem_infos must match batch size"

    spans = []
    for b in range(B):
        spans.append(clause_token_spans_for_row(clean_tokens[b], model, problem_infos[b]))
    return spans




# -----------------
# Attention stats
# -----------------

# --- Attention stats at chosen destination position -----------

@torch.no_grad()
def head_attention_mass_at_pos(
    model,
    clean_tokens: torch.Tensor,                 # [B, S]
    problem_infos: List[Dict[str, str]],        # len B; used if spans_by_label is None
    layer_idx: int,
    head_idx: int,
    dest_pos: int,                              # single integer destination index for all rows
    span_finder=None,                           # kept for compatibility (unused here)
    spans_by_label: Optional[Dict[str, List[Optional[Span]]]] = None,
    normalize: bool = False,
) -> Dict[str, Any]:
    """
    If spans_by_label is None:
      - Behaves like the original: computes spans for 'queried_rule'/'correct_fact' from problem_infos
        and returns 'rule_mass'/'fact_mass' (lists of floats or None).
      - Also returns per-example 'rule_max_other_in_problem' and 'fact_max_other_in_problem', and
        'problem_total_mass' (same total for both).

    If spans_by_label is provided:
      - Use your supplied spans (per label -> per-example Optional[Span]).
      - Returns results under 'by_span' = { label: { 'mass': [...], 'max_other_in_problem': [...] } }
      - Also returns 'problem_total_mass'.
      - For convenience, if labels include 'queried_rule'/'correct_fact', the legacy keys
        'rule_mass'/'fact_mass' and their max-other counterparts are also populated.
    """
    B, S = clean_tokens.shape
    assert isinstance(dest_pos, int), "dest_pos must be a single integer"

    # Run once to get attention patterns (softmax probs): [B, n_heads, dest, src]
    _, cache = model.run_with_cache(clean_tokens)
    patt = cache["pattern", layer_idx][:, head_idx, :, :]  # [B, dest, src]

    # Prepare outputs
    by_span = {}
    problem_total_mass = [None] * B

    # Pre-compute final-problem regions per example
    problem_regions: List[Optional[Span]] = []
    for b in range(B):
        try:
            regions = locate_final_problem_regions(clean_tokens[b], model)
            problem_regions.append(regions["problem_region"])
        except Exception:
            problem_regions.append(None)

    # Compute total attention mass in the final problem, once per sample
    for b in range(B):
        '''if not (0 <= dest_pos < S) or problem_regions[b] is None:
            problem_total_mass[b] = None
            continue'''
        p_s, p_e = problem_regions[b]
        src_all = torch.arange(p_s, p_e, device=patt.device)
        problem_total_mass[b] = float(patt[b, dest_pos, src_all].sum().item())

    # For each label, compute mass on the span and max-other-in-problem
    masses = {}
    for label, _ in spans_by_label.items():
        masses[label] = []
    
    max_others = []
    for b in range(B):
        first_enter = True
        for label, span_list in spans_by_label.items():
            span = span_list[b]
            p_s, p_e = problem_regions[b]
            src_all = torch.arange(p_s, p_e, device=patt.device)

            # Specified span of tokens. Set s1+1 to include trailing tokens such as "." as well.
            s0, s1 = span
            s1 += 1
            # Adjust the span indices if we are specifying the span from the right
            if s0 < 0 and s1 < 0:
                s0 = p_e + s0
                s1 = p_e + s1

            # Attention mass on the specified span of tokens
            src_span = torch.arange(s0, s1, device=patt.device)
            if normalize:
                masses[label].append(float(patt[b, dest_pos, src_span].sum().item()/problem_total_mass[b]))
            else:
                masses[label].append(float(patt[b, dest_pos, src_span].sum().item()))
            by_span[label] = {'mass': masses[label]}

            # Max on other positions within the problem (exclude the span)
            # Build a mask over src_all that zeros out the span window.
            if first_enter:
                mask = torch.ones_like(src_all, dtype=torch.bool)
                first_enter = False
            # indices of the span relative to src_all
            low = max(s0, p_s) - p_s
            high = min(s1, p_e) - p_s
            mask[low:high] = False  # Exclude the (relative) indices in the span

        # CAUTION: max_others is computed at the positions outside of all the spans
        # provided, e.g. if we provide "correct_fact" and "QUERY" token spans, then
        # max_others is obtained at the positions outside of the UNION of those two spans!
        if mask.any():
            if normalize:
                max_others.append(float(patt[b, dest_pos, src_all[mask]].max().item()/problem_total_mass[b]))
            else:
                max_others.append(float(patt[b, dest_pos, src_all[mask]].max().item()))
        else:
            # No "other" tokens remaining, the specified span covers the final problem
            max_others.append(None)

        by_span[label]["max_other_in_problem"] = max_others

    out: Dict[str, Any] = {
        "layer": layer_idx,
        "head": head_idx,
        "dest_pos": dest_pos,
        "problem_total_mass": problem_total_mass,
        "max_other_in_problem": max_others,
        "by_span": by_span,
    }
    return out
