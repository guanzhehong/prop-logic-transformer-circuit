from __future__ import annotations

import itertools
from functools import partial
from typing import Callable, Optional, Sequence, Tuple, Union, overload

import einops
import pandas as pd
import torch
import torch as t
from jaxtyping import Float, Int
from tqdm.auto import tqdm
from typing_extensions import Literal

import transformer_lens.utils as utils
from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.HookedTransformer import HookedTransformer
# Needs some polishing
def get_heads_and_posns_to_keep(
    ctfl_dataset: corrupted_tokens,
    model: HookedTransformer,
    circuit: dict[str, list[tuple[int, int]]],
    seq_pos_to_keep: dict[str, str],
) -> dict[int, Bool[Tensor, "batch seq head"]]:

    heads_and_posns_to_keep = {}
    batch, seq, n_heads = len(ctfl_dataset), len(ctfl_dataset[0]), model.cfg.n_heads

    for layer in range(model.cfg.n_layers):
        mask = t.zeros(size=(batch, seq, n_heads))
        for (head_type, head_list) in circuit.items():
            seq_pos = seq_pos_to_keep[head_type]
            indices = seq_pos
            for (layer_idx, head_idx) in head_list:
                if layer_idx == layer:
                    mask[:, indices, head_idx] = 1
        heads_and_posns_to_keep[layer] = mask.bool()

    return heads_and_posns_to_keep



def hook_fn_mask_z(
    z: Float[Tensor, "batch seq head d_head"],
    hook: HookPoint,
    heads_and_posns_to_keep: dict[int, Bool[Tensor, "batch seq head"]],
    ctfl_actns: Float[Tensor, "layer batch seq head d_head"],
) -> Float[Tensor, "batch seq head d_head"]:
    '''
    Hook function which masks the z output of a transformer head.

    heads_and_posns_to_keep
        dict created with the get_heads_and_posns_to_keep function. This tells
        us where to mask.

    ctfl_actns
        Tensor of counterfactual z values of the ctfl_dataset over each group of prompts
        with the same template. This tells us what values to mask with.
    '''
    # Get the mask for this layer, and add d_head=1 dimension so it broadcasts correctly
    mask_for_this_layer = heads_and_posns_to_keep[hook.layer()].unsqueeze(-1).to(z.device)
    mask_for_this_layer = mask_for_this_layer.to(z.device)
    ctfl_actns = ctfl_actns.to(z.device)

    # Set z values to the mean
    z = t.where(mask_for_this_layer, z, ctfl_actns[hook.layer()])

    return z


def add_ctfl_ablation_hook(
    model: HookedTransformer,
    ctfl_dataset: corrupted_tokens,
    circuit: dict[str, list[tuple[int, int]]],
    seq_pos_to_keep: dict[str, str],
    is_permanent: bool = True,
) -> HookedTransformer:
    '''
    Adds a permanent hook to the model, which ablates according to the circuit and
    seq_pos_to_keep dictionaries.

    In other words, when the model is run on the clean prompts, every head's output will
    be replaced with the mean over ctfl_dataset for sequences with the same template,
    except for a subset of heads and sequence positions as specified by the circuit
    and seq_pos_to_keep dicts.
    '''
    # Prevent pathological behaviors with hooks
    model.reset_hooks(including_permanent=True)

    # Cache the outputs of every head
    _, ctfl_cache = model.run_with_cache(
        ctfl_dataset,
        return_type=None,
        names_filter=lambda name: name.endswith("z"),
    )
    # Create tensor to store counterfactual activations
    n_layers, n_heads, d_head = model.cfg.n_layers, model.cfg.n_heads, model.cfg.d_head
    batch, seq_len = len(ctfl_dataset), len(ctfl_dataset[0])
    ctfl_actns = torch.zeros(size=(n_layers, batch, seq_len, n_heads, d_head), device=model.cfg.device)

    # Get set of different templates for this data
    for layer in range(model.cfg.n_layers):
        z_for_this_layer = ctfl_cache[utils.get_act_name("z", layer)] # [batch seq head d_head]

        ctfl_actns[layer] = z_for_this_layer

    ### End computing counterfactual activations

    heads_and_posns_to_keep = get_heads_and_posns_to_keep(ctfl_dataset, model, circuit, seq_pos_to_keep)

    # Get a hook function which will patch in the mean z values for each head, at
    # all positions which don't belong to the circuit
    hook_fn = partial(
        hook_fn_mask_z,
        heads_and_posns_to_keep=heads_and_posns_to_keep,
        ctfl_actns=ctfl_actns
    )

    # Apply hook
    model.add_hook(lambda name: name.endswith("z"), hook_fn, is_permanent=is_permanent)

    return model
