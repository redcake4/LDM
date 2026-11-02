"""Account for the gap between the built model and the 62.9M reported in the paper.

Paper: arXiv:2609.32794v1, Section 3 "Implementation" reports 62.9M parameters and
1.72 TFLOPs per forward pass for a 10-block, width-512 LSF at p=2.

This script reports the exact breakdown of the released model and the parameter cost
of each component that docs/paper_alignment.md lists as missing, so that the
remaining difference can be attributed rather than guessed.

Usage:
    python scripts/param_audit.py [--config configs/paper_lsf_t1n_t1c.yaml]
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ldm.config import load_config
from ldm.model import LDMModel3D

PAPER_PARAMS = 62.9e6


def count(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def candidates(hidden, depth, mlp_ratio, local_dim, bottleneck_dim, patch_size, in_chans=8):
    """Parameter cost of each component the paper describes but the code omits."""
    linear = lambda i, o, bias=True: i * o + (o if bias else 0)

    # A: condition-side patch embedding required by Eq. (3).
    cond_embed = (linear(in_chans * patch_size ** 3, bottleneck_dim, bias=False)
                  + linear(bottleneck_dim, hidden))

    # B: time-modulated RMSNorm on the fused mixer output, per block.
    post_fusion = (hidden + linear(hidden, 2 * hidden)) * depth

    # C: SwiGLU hidden width without the 2/3 reduction applied in ldm/layers.py.
    narrow = int(int(hidden * mlp_ratio) * 2 / 3)
    wide = int(hidden * mlp_ratio)
    swiglu_delta = ((linear(hidden, 2 * wide) + linear(wide, hidden))
                    - (linear(hidden, 2 * narrow) + linear(narrow, hidden))) * depth

    # D: separate depthwise kernels for dilations {1, 2, 4} instead of one shared kernel.
    dilations = local_dim * 27 * 2 * depth

    # E: per-branch output projections instead of one projection over the concatenation.
    per_branch = (linear(hidden, hidden) + linear(local_dim, hidden)
                  - linear(hidden + local_dim, hidden)) * depth

    return {
        "A condition-side patch embedding (Eq. 3)": cond_embed,
        "B post-fusion time-modulated RMSNorm": post_fusion,
        "C SwiGLU width without the 2/3 reduction": swiglu_delta,
        "D separate kernels per dilation {1,2,4}": dilations,
        "E per-branch output projections (Eq. 4/5)": per_branch,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/paper_lsf_t1n_t1c.yaml")
    parser.add_argument("--json", help="Write the audit to this path")
    args = parser.parse_args()

    config = load_config(args.config)
    model = LDMModel3D(**config["model"])
    m = config["model"]
    total = count(model)

    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    pos = model.pos_embed.numel()
    cond = 0 if model.c_embedder is None else sum(p.numel() for p in model.c_embedder.parameters())

    print(f"config {args.config}")
    print(f"trainable {total/1e6:9.3f}M")
    print(f"frozen    {frozen/1e6:9.3f}M  = sincos pos_embed {pos/1e6:.3f}M "
          f"+ condition embedder {cond/1e6:.3f}M")
    print(f"all       {(total+frozen)/1e6:9.3f}M")
    print(f"paper     {PAPER_PARAMS/1e6:9.3f}M")
    print(f"gap vs trainable {(PAPER_PARAMS-total)/1e6:+.3f}M   "
          f"gap vs all {(PAPER_PARAMS-total-frozen)/1e6:+.3f}M")
    print("   (the paper does not say whether its 62.9M includes the non-trainable\n"
          "    positional embedding; neither reading matches)\n")

    print("released breakdown")
    block = model.blocks[0]
    parts = [("x_embedder", count(model.x_embedder)),
             ("t_embedder", count(model.t_embedder)),
             ("final_layer", count(model.final_layer)),
             ("blocks (x%d)" % len(model.blocks), count(model.blocks))]
    for name, value in parts:
        print(f"   {name:24s} {value/1e6:8.3f}M")
    print("   one block")
    for name, child in [("mixer", block.mixer), ("mlp", block.mlp),
                        ("adaLN_modulation", block.adaLN_modulation),
                        ("mdsa gates", block.mdsa)]:
        if child is not None:
            print(f"      {name:21s} {count(child)/1e6:8.3f}M")

    deltas = candidates(m["hidden_size"], m["depth"], m["mlp_ratio"],
                        m["local_dim"], m["bottleneck_dim"], m["patch_size"])
    # Components already enabled by this configuration are not missing from it.
    if m["post_fusion_norm"]:
        deltas.pop("B post-fusion time-modulated RMSNorm", None)
    if m["mdsa_basis"] != "state":
        deltas.pop("A condition-side patch embedding (Eq. 3)", None)
    if len(m["local_dilations"]) > 1:
        deltas.pop("D separate kernels per dilation {1,2,4}", None)
    print("\ncost of each component the paper describes but this config still omits")
    for name, value in deltas.items():
        print(f"   {name:46s} {value/1e6:+8.3f}M")

    print("\ncombinations within 0.5M of the paper's 62.9M")
    names = list(deltas)
    hits = []
    for size in range(1, len(names) + 1):
        for combo in itertools.combinations(names, size):
            reached = total + sum(deltas[c] for c in combo)
            if abs(reached - PAPER_PARAMS) <= 0.5e6:
                hits.append((combo, reached))
                print(f"   {reached/1e6:7.3f}M  <- {' + '.join(c[0] for c in combo)}")
    if not hits:
        best = min(
            (itertools.combinations(names, size) for size in range(1, len(names) + 1)),
            key=lambda _: 0)
        print("   none")
        print("\n   closest single components:")
        for name, value in sorted(deltas.items(), key=lambda kv: abs(total + kv[1] - PAPER_PARAMS)):
            print(f"   {(total+value)/1e6:7.3f}M  <- {name}")
        print("\n   No combination of the documented omissions reaches 62.9M.")
        print("   The released block therefore differs from the paper's block in a way")
        print("   that is not yet documented. The authors should publish the exact")
        print("   block definition used for the reported results.")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "config": args.config, "built": total, "paper": PAPER_PARAMS,
            "gap": PAPER_PARAMS - total,
            "candidates": deltas,
            "combinations_within_0_5M": [
                {"components": list(c), "total": r} for c, r in hits],
        }, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
