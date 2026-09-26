"""Module 10C: Convolutional Code (Rate 1/2, K=3), Viterbi Decoding & Blind Identification.

Scientific Status: 🔒 LOCKED (Controlled Candidate Pool).
Source: # Module 10 — FEC, CRC & Interleaver Analysis, Section 9.

Parameters:
    Code Rate: R = 1/2
    Constraint Length: K = 3 (4 states)
    Generators: G1 = 111_2 (7 octal), G2 = 101_2 (5 octal)
    State Convention:
        state_t = (u[t-1], u[t-2])
        next_state = (u[t], u[t-1])

Deterministic Validation:
    Input: [1, 0, 1, 1, 0]
    Encoded Sequence: [1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0]

Viterbi Decoding Characteristics:
    - 0% BER -> Decoded BER = 0.0, Frame Success = 1.000
    - 1% BER -> Raw 0.0098 -> Decoded 0.0, Frame Success = 0.998
    - 5% BER -> Raw 0.0489 -> Decoded 0.0067, Frame Success = 0.758
    - 10% BER -> Raw 0.0986 -> Decoded 0.0609, Frame Success = 0.176
    - 20% BER -> Raw 0.1999 -> Decoded 0.2815 (worse than raw!)
    - 30% BER -> Raw 0.2992 -> Decoded 0.4271

Candidate Generator Identification:
    Received Bits -> Candidate Viterbi -> Decoded Information -> Re-encode -> Reconstruction Mismatch.
    Known K=3 pool: Top-1 = 1.00 (0-10% BER), ~0.915 at 20% BER.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


CONV_CANDIDATE_POOL_K3 = [
    {"name": "CONV_K3_G7_G5", "g1": [1, 1, 1], "g2": [1, 0, 1]},
    {"name": "CONV_K3_G7_G7", "g1": [1, 1, 1], "g2": [1, 1, 1]},
    {"name": "CONV_K3_G5_G7", "g1": [1, 0, 1], "g2": [1, 1, 1]},
    {"name": "CONV_K3_G5_G5", "g1": [1, 0, 1], "g2": [1, 0, 1]}
]


def encode_convolutional_k3(
    info_bits: Union[np.ndarray, List[int]],
    g1: Optional[List[int]] = None,
    g2: Optional[List[int]] = None,
    flush_tail: bool = True
) -> np.ndarray:
    """Encode binary message bits with rate 1/2, K=3 convolutional code.

    State: (u[t-1], u[t-2])
    G1: [1, 1, 1] -> y1 = u[t] ^ u[t-1] ^ u[t-2]
    G2: [1, 0, 1] -> y2 = u[t] ^ u[t-2]
    """
    u = np.asarray(info_bits, dtype=np.uint8).flatten()
    gen1 = g1 if g1 is not None else [1, 1, 1]
    gen2 = g2 if g2 is not None else [1, 0, 1]

    # Append 2 zero tail bits to flush state back to 00 if requested
    stream = np.concatenate([u, np.zeros(2, dtype=np.uint8)]) if flush_tail else u
    output_bits = []

    m1, m2 = 0, 0  # Initial state (u[t-1], u[t-2]) = (0, 0)
    for b in stream:
        y1 = (b * gen1[0] ^ m1 * gen1[1] ^ m2 * gen1[2]) & 1
        y2 = (b * gen2[0] ^ m1 * gen2[1] ^ m2 * gen2[2]) & 1
        output_bits.extend([y1, y2])
        m2 = m1
        m1 = int(b)

    return np.array(output_bits, dtype=np.uint8)


def viterbi_decode_hard_k3(
    rx_bits: Union[np.ndarray, List[int]],
    g1: Optional[List[int]] = None,
    g2: Optional[List[int]] = None,
    flush_tail: bool = True
) -> Tuple[np.ndarray, float]:
    """Perform hard-decision Viterbi decoding over 4-state trellis for rate 1/2, K=3 code.

    Returns:
        Tuple of (decoded_information_bits, normalized_path_metric).
    """
    rx = np.asarray(rx_bits, dtype=np.uint8).flatten()
    N_pairs = len(rx) // 2
    if N_pairs == 0:
        return np.empty(0, dtype=np.uint8), 0.0

    gen1 = g1 if g1 is not None else [1, 1, 1]
    gen2 = g2 if g2 is not None else [1, 0, 1]

    # Trellis states: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # State mapping: state = (m1 << 1) | m2
    num_states = 4
    INF = 1e9
    path_metrics = np.full(num_states, INF, dtype=np.float64)
    path_metrics[0] = 0.0  # Start in state (0,0)

    # Precompute branch outputs for (state, input_bit)
    # branch_out[state, input] = (y1, y2)
    # next_state[state, input] = new_state
    branch_out = np.zeros((num_states, 2, 2), dtype=np.uint8)
    next_state = np.zeros((num_states, 2), dtype=np.int32)
    for s in range(num_states):
        m1 = (s >> 1) & 1
        m2 = s & 1
        for u in [0, 1]:
            y1 = (u * gen1[0] ^ m1 * gen1[1] ^ m2 * gen1[2]) & 1
            y2 = (u * gen2[0] ^ m1 * gen2[1] ^ m2 * gen2[2]) & 1
            branch_out[s, u] = [y1, y2]
            next_state[s, u] = ((u << 1) | m1) & 3

    # History: history[t, state] = (prev_state, input_bit)
    history = np.zeros((N_pairs, num_states, 2), dtype=np.int32)

    for t in range(N_pairs):
        pair = rx[2 * t : 2 * t + 2]
        new_metrics = np.full(num_states, INF, dtype=np.float64)

        for s in range(num_states):
            if path_metrics[s] >= INF:
                continue
            for u in [0, 1]:
                ns = next_state[s, u]
                out = branch_out[s, u]
                bm = int(out[0] ^ pair[0]) + int(out[1] ^ pair[1])
                cand_metric = path_metrics[s] + bm
                if cand_metric < new_metrics[ns]:
                    new_metrics[ns] = cand_metric
                    history[t, ns] = [s, u]

        path_metrics = new_metrics

    # Traceback: if flush_tail, end in state 0, else best metric state
    best_state = 0 if flush_tail else int(np.argmin(path_metrics))
    best_metric = float(path_metrics[best_state]) if path_metrics[best_state] < INF else 0.0

    decoded_rev = []
    curr_state = best_state
    for t in range(N_pairs - 1, -1, -1):
        prev_s, u = history[t, curr_state]
        decoded_rev.append(u)
        curr_state = prev_s

    decoded_bits = np.array(decoded_rev[::-1], dtype=np.uint8)
    # Remove tail bits if flushed
    if flush_tail and len(decoded_bits) >= 2:
        decoded_info = decoded_bits[:-2]
    else:
        decoded_info = decoded_bits

    norm_metric = float(best_metric / (2 * N_pairs)) if N_pairs > 0 else 0.0
    return decoded_info, norm_metric


def identify_convolutional_generator(
    rx_bits: Union[np.ndarray, List[int]],
    candidate_generators: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Rank candidate convolutional generators using Viterbi re-encoding reconstruction mismatch.

    Args:
        rx_bits: 1D array of received bits.
        candidate_generators: List of generator candidates (default: CONV_CANDIDATE_POOL_K3).

    Returns:
        Structured dictionary containing ranked candidates, best mismatch, and scientific status.
    """
    rx = np.asarray(rx_bits, dtype=np.uint8).flatten()
    cands = candidate_generators if candidate_generators is not None else CONV_CANDIDATE_POOL_K3

    results = []
    for cand in cands:
        g1, g2 = cand["g1"], cand["g2"]
        decoded, path_m = viterbi_decode_hard_k3(rx, g1=g1, g2=g2, flush_tail=True)
        # Re-encode decoded sequence
        re_encoded = encode_convolutional_k3(decoded, g1=g1, g2=g2, flush_tail=True)

        n_comp = min(len(rx), len(re_encoded))
        if n_comp > 0:
            mismatches = int(np.sum(rx[:n_comp] ^ re_encoded[:n_comp]))
            mismatch_rate = float(mismatches / n_comp)
        else:
            mismatch_rate = 1.0

        results.append({
            "candidate": cand["name"],
            "reconstruction_mismatch": mismatch_rate,
            "path_metric": path_m,
            "decoded_bits_count": len(decoded)
        })

    results.sort(key=lambda x: x["reconstruction_mismatch"])
    best = results[0] if results else None

    return {
        "status": "SUCCESS",
        "best_candidate": best["candidate"] if best else None,
        "best_mismatch": best["reconstruction_mismatch"] if best else 1.0,
        "candidates": results,
        "scientific_status": "LOCKED (Controlled Candidate Pool, Section 9.3)",
        "safeguards": (
            "At BER >= 20%, Viterbi decoding degrades below raw BER. "
            "Reconstruction mismatch identifies candidate generators within tested pools."
        )
    }
