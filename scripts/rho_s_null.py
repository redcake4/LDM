"""Monte-Carlo null distribution of the structural fraction rho_S.

Settles the inconsistency between the two Gaussian reference values quoted in
arXiv:2609.32794v1:

  * Table 4 uses r=2 and quotes 6.25 %, which is exactly r/K = 2/32.
  * Table 1(B) uses r=8 (Section 3.1) and reports 33-34 %, where the same
    formula gives 8/32 = 25 %.

Manuscript Eq. (6) defines rho_S with the projector P* taken from the *target*
latent z_t. For an error delta that is independent of z_t, that projector is
fixed with respect to delta, so E[rho_S] = r/K exactly, regardless of the
distribution of z. An adaptive basis -- one estimated from the perturbed latent
z+delta, or from delta itself -- selects the top r of K *sample* eigendirections
and therefore captures strictly more than r/K.

This script measures all three conventions on pure Gaussian tensors. It needs no
data, no autoencoder and no GPU.

Usage:
    python scripts/rho_s_null.py
    python scripts/rho_s_null.py --ranks 1 2 4 8 16 --trials 4000 --json out.json
"""
import argparse
import json
import math
from pathlib import Path

import torch


def top_r_projector(matrix, rank):
    """P = U U^T from the top-`rank` eigenvectors of the window Gram matrix (Eq. (1))."""
    gram = torch.matmul(matrix, matrix.transpose(-2, -1))
    _, vectors = torch.linalg.eigh(gram)
    basis = vectors[..., -rank:]
    return torch.matmul(basis, basis.transpose(-2, -1))


def structural_fraction(projector, error):
    """rho_S = ||P delta||_F^2 / ||delta||_F^2, summed over windows (Eq. (6))."""
    projected = torch.matmul(projector, error)
    return (projected.square().sum((-2, -1)) / error.square().sum((-2, -1)))


def run(rank, tokens, features, trials, noise_ratio, generator, dtype=torch.float64):
    shape = (trials, tokens, features)
    latent = torch.randn(shape, generator=generator, dtype=dtype)
    error = torch.randn(shape, generator=generator, dtype=dtype)
    # Match the manuscript's ||delta|| = noise_ratio * ||z||, per window.
    scale = noise_ratio * latent.norm(dim=(-2, -1), keepdim=True) / error.norm(dim=(-2, -1), keepdim=True)
    error = error * scale

    out = {}
    out["basis from z (Eq. 6 as written)"] = structural_fraction(
        top_r_projector(latent, rank), error)
    out["basis from z + delta (adaptive)"] = structural_fraction(
        top_r_projector(latent + error, rank), error)
    out["basis from delta (fully adaptive)"] = structural_fraction(
        top_r_projector(error, rank), error)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ranks", type=int, nargs="+", default=[2, 8])
    parser.add_argument("--tokens", type=int, default=32,
                        help="K, tokens per window; the paper uses 2*4*4 = 32")
    parser.add_argument("--features", type=int, default=512,
                        help="feature width of a window row")
    parser.add_argument("--trials", type=int, default=2000)
    parser.add_argument("--noise-ratio", type=float, default=0.1,
                        help="||delta|| / ||z||; the paper uses 0.1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json")
    args = parser.parse_args()

    generator = torch.Generator().manual_seed(args.seed)
    K = args.tokens
    print(f"K = {K} tokens per window, {args.features} features, "
          f"{args.trials} windows, ||delta|| = {args.noise_ratio}||z||\n")

    report = {"tokens": K, "features": args.features, "trials": args.trials,
              "noise_ratio": args.noise_ratio, "seed": args.seed, "ranks": {}}

    for rank in args.ranks:
        if not 1 <= rank <= K:
            raise SystemExit(f"rank {rank} must lie in [1, {K}]")
        theory = rank / K
        print(f"rank r = {rank}   theoretical r/K = {100*theory:.2f}%")
        results = run(rank, K, args.features, args.trials, args.noise_ratio, generator)
        entry = {"theory_r_over_K": theory, "conventions": {}}
        for name, values in results.items():
            mean = float(values.mean()) * 100
            sd = float(values.std()) * 100
            stderr = sd / math.sqrt(args.trials)
            flag = "matches r/K" if abs(mean - 100 * theory) < 4 * stderr else "DIFFERS from r/K"
            print(f"   {name:38s} {mean:6.2f}% +- {sd:4.2f}   ({flag})")
            entry["conventions"][name] = {"mean_percent": mean, "sd_percent": sd}
        report["ranks"][str(rank)] = entry
        print()

    print("Reading of the manuscript:")
    for rank, quoted in ((2, 6.25), (8, 33.5)):
        if str(rank) in report["ranks"]:
            fixed = report["ranks"][str(rank)]["conventions"][
                "basis from z (Eq. 6 as written)"]["mean_percent"]
            adaptive = report["ranks"][str(rank)]["conventions"][
                "basis from z + delta (adaptive)"]["mean_percent"]
            print(f"   r={rank}: paper quotes ~{quoted}%; Eq. (6) as written gives "
                  f"{fixed:.2f}%, an adaptive basis gives {adaptive:.2f}%")

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
