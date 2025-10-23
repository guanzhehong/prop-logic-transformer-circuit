from __future__ import annotations

import itertools
from functools import partial
from typing import Callable, Optional, Sequence, Tuple, Union, overload

import einops
import pandas as pd
import torch
from jaxtyping import Float, Int
from tqdm.auto import tqdm
from typing_extensions import Literal

import transformer_lens.utils as utils
from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.HookedTransformer import HookedTransformer

# This activation patching workflow is built on the Arena IOI implementation:
# https://arena-chapter1-transformer-interp.streamlit.app/[1.4.1]_Indirect_Object_Identification

# ------------------------
# Basic tools and metrics
# ------------------------
def logits_diff(
    logits, answer_tokens, per_prompt = False
):
    '''
    Returns logit difference between the clean and corrupt answer. If per_prompt=True, 
    return the array of differences rather than the average.

    A small note on terminology: in our experiments, the clean and corrupt problem-answer pairs are
    both "logically sensible". For example, in the QUERY-based patching experiments, for the clean
    and corrupt problem-answer pairs, one queries the Logical-Operator chain, while another queries
    the linear chain: both have logically valid answers and proofs for getting to the answer. 
    We may also use "normal vs. alternative" to describe the contrastive differences.
    
    This is unlike the classic IOI problem, where the corrupt problem-answer pair from the 
    so-called "ABC" dataset does not have a (logically) sensible solution.
    '''
    # Only the last-position logits are relevant for predicting the answer token
    final_logits = logits[:, -1, :].cpu()
    final_logits = final_logits.cpu()
    answer_tokens = answer_tokens.cpu()
    answer_logits = final_logits.gather(dim=-1, index=answer_tokens).cpu()
    correct_logits, incorrect_logits = answer_logits.unbind(dim=-1)
    answer_logit_diff = correct_logits - incorrect_logits
    return answer_logit_diff if per_prompt else answer_logit_diff.mean()


def basic_metric(
    logits, corrupted_logit_diff, clean_logit_diff, answer_tokens, normalize = True, average = True
):
    """ Calibrated logit-difference metric, very similar to the classic IOI implementation. 
    The closer the value is to 1, the closer the patched run's behavior 
    (assessed via logit difference between clean and corrupt answers)
    is to the corrupt run. OTOH, the closer to 0, the weaker the intervention's effects 
    on the logit difference. """
    if average:
        patched_logit_diff = logits_diff(logits, answer_tokens)
        if normalize:
            return (patched_logit_diff - clean_logit_diff) / (corrupted_logit_diff  - clean_logit_diff)
        else:
            return patched_logit_diff - clean_logit_diff
    else:
        # When we want to preserve sample-wise shifts in logit difference, we will 
        # pass the sample-wise clean logit difference into this function, and return
        # the array of results, instead of a single value.
        patched_logit_diff = logits_diff(logits, answer_tokens, per_prompt=True)
        return patched_logit_diff - clean_logit_diff


# ---------------------------------------
# Activation Patching of Attention Heads
# ---------------------------------------
def patch_head(
    clean_head_vector: Float[Tensor, "batch pos head_index d_head"],
    hook: HookPoint,
    head_index: int,
    corrupted_cache: ActivationCache,
    positions_l: int,
    positions_u: int,
) -> Float[Tensor, "batch pos head_index d_head"]:
    '''
    Patches the output of a given head (before it's added to the residual stream) at
    every sequence position, using the value from the clean cache.
    '''
    if positions_u == 0:
        clean_head_vector[:, positions_l:, head_index] = corrupted_cache[hook.name][:, positions_l:, head_index]
    else:
        clean_head_vector[:, positions_l:positions_u, head_index] = corrupted_cache[hook.name][:, positions_l:positions_u, head_index]
    return clean_head_vector


def basic_patching(
    model: HookedTransformer,
    clean_tokens: Float[Tensor, "batch pos"],
    clean_logit_diff: Float[Tensor, "batch seq d_vocab"],
    corrupted_logit_diff: Float[Tensor, "batch seq d_vocab"],
    corrupted_cache: ActivationCache,
    metric: Callable,
    component: str,
    answer_tokens_batch: Float[Tensor, "batch 2"],
    GQA_constant: int,
    positions_l: int,
    positions_u: int,
    batch_size: int,
) -> Float[Tensor, "layer head"]:
    '''
    Returns an array of patching results at the selected positions specified by 
    [positions_l, positions_u] for every head in the model (using corrupted cache).
    
    `component` choices: ["z", "k", "q", "v"]. For key and value ("k" and "v"), note that
    current LLMs often use grouped-query attention (e.g. Gemma-2 family has each 2 heads 
    sharing the same k and v activations, while Mistral-7B-v0.1 uses 4), so we need
    to specify GQA_constant in the input to properly handle the results' shapes.
    '''
    patched_logit_diff_history = torch.zeros(len(component), model.cfg.n_layers, model.cfg.n_heads, batch_size, device="cuda", dtype=torch.float32)

    for component_idx, component in enumerate(component):
        if component == "q" or component == "z":
          upper_limit = model.cfg.n_heads
        else:
          upper_limit = int(model.cfg.n_heads / GQA_constant)
        
        print("Sweeping component " + str(component)) #model.cfg.n_layers
        for (layer, head) in tqdm(list(itertools.product(range(model.cfg.n_layers), range(upper_limit)))):
                hook_fn_general = patch_head
                hook_fn = partial(hook_fn_general, 
                                  head_index = head, 
                                  corrupted_cache = corrupted_cache,
                                  positions_l = positions_l,
                                  positions_u = positions_u)
                patched_logits = model.run_with_hooks(
                    clean_tokens,
                    fwd_hooks = [(utils.get_act_name(component, layer), hook_fn)],
                    return_type="logits"
                )
                patched_logit_diff_history[component_idx, layer, head] = metric(patched_logits, 
                                                                                corrupted_logit_diff, 
                                                                                clean_logit_diff, 
                                                                                answer_tokens = answer_tokens_batch)
    return patched_logit_diff_history