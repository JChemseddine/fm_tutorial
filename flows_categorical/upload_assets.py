"""Bundle the checkpoints from `networks/` into a HuggingFace Hub layout and push.

Notebook `notebooks/05_continuous_discrete_sudoku.ipynb` falls back to HF Hub
`Jugc/fm-tutorial` when local `networks/` checkpoints are missing. This
script prepares + pushes that HF repo.

Local layout (input):

    networks/
    ├── vmf_d11_p1/final.pt
    ├── vmf_tc_d11_p1/final.pt
    └── masked_p1/final.pt

HF Hub layout (output):

    Jugc/fm-tutorial/
    ├── vmf_d11_p1/checkpoint.pt
    ├── vmf_tc_d11_p1/checkpoint.pt
    └── masked_p1/checkpoint.pt

Each `final.pt` already contains the training config inline (under the
`"config"` key), so no separate `config.json` is needed. The held-out puzzles
ship in the repo at `data/sudoku_extreme_100.npz` and are NOT uploaded.

Usage (run from the fm_tutorial repo root):

    # Stage the bundle locally
    python flows_categorical/upload_assets.py bundle

    # Push to HF Hub (requires `huggingface-cli login` first)
    python flows_categorical/upload_assets.py push
"""

import argparse
import shutil
from pathlib import Path


METHOD_NAMES = ("vmf_d11_p1", "vmf_tc_d11_p1", "masked_p1")


def cmd_bundle(args):
    src = Path(args.networks_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in METHOD_NAMES:
        sub_src = src / name / "final.pt"
        if not sub_src.exists():
            raise FileNotFoundError(f"missing {sub_src}")
        sub_out = out / name
        sub_out.mkdir(exist_ok=True)
        shutil.copy(sub_src, sub_out / "checkpoint.pt")
        size_mb = (sub_out / "checkpoint.pt").stat().st_size / 1024 / 1024
        print(f"  bundled {name:18s} -> {sub_out / 'checkpoint.pt'}  ({size_mb:.0f} MB)")
    (out / "README.md").write_text(
        "# fm-tutorial assets\n\n"
        "Pretrained Sudoku checkpoints for "
        "[fm_tutorial notebook 05](https://github.com/JChemseddine/fm_tutorial/"
        "blob/main/notebooks/05_continuous_discrete_sudoku.ipynb).\n\n"
        "Three 28M-parameter DiT checkpoints on *Sudoku Extreme*:\n"
        "- `vmf_d11_p1/checkpoint.pt` — spherical flow matching (vMF), no time conditioning\n"
        "- `vmf_tc_d11_p1/checkpoint.pt` — same, with time conditioning\n"
        "- `masked_p1/checkpoint.pt` — masked diffusion baseline (MDLM-style)\n\n"
        "Each `.pt` is a torch.save dict with keys `model_state_dict`, "
        "`ema_state_dict`, `config`, `step`, and (for vmf) `warp_state`.\n\n"
        "Source: *Spherical Flows for Sampling Discrete Distributions* "
        "([arXiv:2605.05629](https://arxiv.org/abs/2605.05629)).\n"
    )
    print(f"Bundle ready at {out}/")


def cmd_push(args):
    from huggingface_hub import HfApi, create_repo
    create_repo(args.repo, exist_ok=True, repo_type="model")
    api = HfApi()
    api.upload_folder(
        folder_path=args.bundle_dir,
        repo_id=args.repo,
        repo_type="model",
        commit_message="Upload fm_tutorial Sudoku checkpoints",
    )
    print(f"Pushed {args.bundle_dir} -> https://huggingface.co/{args.repo}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(required=True, dest="cmd")

    b = sub.add_parser("bundle", help="Stage checkpoints into out_dir/ with HF layout")
    b.add_argument("--networks-dir", default="./networks",
                   help="Directory containing <method>/final.pt subdirs (default: ./networks)")
    b.add_argument("--out-dir", default="./hf_bundle")
    b.set_defaults(func=cmd_bundle)

    pu = sub.add_parser("push", help="Push a staged bundle to HF Hub")
    pu.add_argument("--bundle-dir", default="./hf_bundle")
    pu.add_argument("--repo", default="Jugc/fm-tutorial")
    pu.set_defaults(func=cmd_push)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
