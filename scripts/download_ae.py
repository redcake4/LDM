"""Download and verify the pinned official MAISI AE; never overwrite a mismatched asset."""
import argparse
from pathlib import Path
import os
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ldm.autoencoder import AE_REPO, AE_REVISION, AE_FILENAME, AE_SHA256, verify_ae
from ldm.config import resolve_path
from ldm.runtime import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="ae_assets/maisi_v1/autoencoder_v1.pt")
    args = parser.parse_args()
    path = resolve_path(args.output)
    if not path.exists():
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError("Install huggingface-hub, or use the pinned direct download in README") from exc
        downloaded = Path(hf_hub_download(repo_id=AE_REPO, filename=AE_FILENAME, revision=AE_REVISION))
        verify_ae(downloaded)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".download")
        shutil.copyfile(downloaded, temporary)
        verify_ae(temporary)
        os.replace(temporary, path)
    verify_ae(path)
    write_json(path.with_name(path.name + ".provenance.json"), {"repo_id": AE_REPO, "filename": AE_FILENAME,
        "revision": AE_REVISION, "sha256": AE_SHA256, "status": "verified"})
    print(f"Verified MAISI AE: {path}")


if __name__ == "__main__":
    main()
