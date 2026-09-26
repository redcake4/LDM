import hashlib
import json
import re
import struct
import xml.etree.ElementTree as ET
from ldm.config import ROOT


def test_manuscript_framework_is_the_original_download():
    raw = (ROOT / "assets/framework.png").read_bytes()
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    assert struct.unpack(">II", raw[16:24]) == (2759, 686)
    assert hashlib.sha256(raw).hexdigest() == "078870c8075089ff69235a7283e558bd4d13bb036e78122246f733031d1447e8"


def test_reference_image_identity_and_scope():
    directory = ROOT / "assets/reference"
    manifest = json.loads((directory / "case06_provenance.json").read_text(encoding="utf-8"))
    raw = (directory / manifest["figure"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == manifest["png_sha256"]
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    assert list(struct.unpack(">II", raw[16:24])) == manifest["png_dimensions"]
    assert manifest["new_ldm_results"] is False
    assert manifest["public_release_ready"] is False
    assert manifest["method_labels_match_all_sources"] is False
    assert len(manifest["cases"]) == 2
    assert sum(len(case["predictions"]) for case in manifest["cases"]) == 10
    assert all(not p["is_current_ldm_d1_result"] for c in manifest["cases"] for p in c["predictions"])


def test_public_reference_manifest_has_no_private_paths_or_ids():
    raw = (ROOT / "assets/reference/case06_provenance.json").read_text(encoding="utf-8")
    assert "subject_id" not in raw and "BraTS-GLI-" not in raw
    assert "/root/" not in raw
    assert re.search(r"(?<![A-Za-z])[A-Z]:[\\/]", raw, flags=re.I) is None


def test_visualization_links_resolve():
    for name in ("README.md",):
        path = ROOT / name
        text = path.read_text(encoding="utf-8")
        links = re.findall(r"\]\(([^)]+)\)", text)
        links += re.findall(r'(?:src|href)="([^"]+)"', text)
        for link in links:
            if not link.startswith(("http://", "https://", "#")):
                target, _, anchor = link.partition("#")
                linked = path.parent / target
                assert linked.exists(), f"Broken documentation link: {name}: {link}"
                if anchor and linked.suffix == ".md":
                    headings = re.findall(r"^#{1,6} (.+)$", linked.read_text(encoding="utf-8"), re.M)
                    anchors = {re.sub(r"[^\w -]", "", title.lower()).replace(" ", "-") for title in headings}
                    assert anchor in anchors, f"Broken documentation anchor: {name}: {link}"


def test_readme_is_a_concise_single_document_workflow():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    commands = re.findall(r"```bash\n(.*?)\n```", readme, re.S)
    assert len(commands) == 3
    assert commands[0].splitlines() == [
        "python -m pip install -r requirements.txt",
        "python scripts/check_dataset.py --task t1n_t1c --scan",
        "python scripts/download_ae.py",
        "python -u precompute_latents.py --task t1n_t1c --device cuda --resume",
    ]
    assert commands[1] == "python -u train.py --config configs/t1n_t1c_p4.yaml --device cuda"
    assert len(commands[2].splitlines()) == 2
    assert commands[2].splitlines()[0].startswith("python -u export_predictions.py --run-dir ")
    assert commands[2].splitlines()[1].startswith("python -u evaluate.py --task t1n_t1c ")
    assert re.findall(r"^## (.+)$", readme, re.M) == ["Data Preparation", "Run", "Evaluation", "Acknowledgements"]
    assert '<a id="acknowledgements"></a>' in readme
    assert len(readme.splitlines()) <= 90
    for name in ("data/h5/t1n__t1c_3d.h5", "data/h5/t2w__t2f_3d.h5",
                 "ae_assets/maisi_v1/autoencoder_v1.pt", "data/latents/maisi_v1/",
                 "LICENSES/ViTTT_DiT_LICENSE.txt", "docs/extracted_sources.json"):
        assert name in readme
    assert "Google Drive download link will be added" in readme


def test_only_root_readme_is_published():
    markdown = sorted(path.relative_to(ROOT).as_posix() for path in ROOT.rglob("*.md")
                      if not any(part.startswith(".") for part in path.relative_to(ROOT).parts))
    assert markdown == ["README.md"]
    for name in ("data/h5/.gitkeep", "data/latents/.gitkeep", "ae_assets/maisi_v1/.gitkeep"):
        assert (ROOT / name).read_bytes().strip() == b""


def test_ten_case_gallery_identity_and_scope():
    directory = ROOT / "assets/qualitative"
    text = (directory / "manifest.json").read_text(encoding="utf-8")
    manifest = json.loads(text)
    assert manifest["image_count"] == 10
    assert manifest["unique_cases"] == {"t1n_t1c": 10, "t2w_t2f": 1}
    assert manifest["new_ldm_results"] is False
    assert manifest["public_release_ready"] is False
    assert manifest["method_labels_match_all_sources"] is False
    assert manifest["checks"]["raw_volume_metrics_recomputed_this_pass"] is False
    assert manifest["checks"]["stored_mse_psnr_arithmetic_checks"] == 100
    assert "BraTS-GLI-" not in text and "/root/" not in text
    assert re.search(r"(?<![A-Za-z])[A-Z]:[\\/]", text, re.I) is None
    assert [item["case"] for item in manifest["images"]] == list(range(1, 11))
    assert len(list(directory.glob("*.png"))) == 10
    for item in manifest["images"]:
        raw = (directory / item["file"]).read_bytes()
        assert item["file"] == f"case_{item['case']:02d}.png"
        assert hashlib.sha256(raw).hexdigest() == item["sha256"]
        assert len(raw) == item["size_bytes"]
        assert raw[:8] == b"\x89PNG\r\n\x1a\n"
        assert list(struct.unpack(">II", raw[16:24])) == item["dimensions"]
        assert [task["task"] for task in item["tasks"]] == ["t1n_t1c", "t2w_t2f"]
        assert sum(len(task["predictions"]) for task in item["tasks"]) == 10
        assert all(not p["is_current_ldm_result"] for task in item["tasks"] for p in task["predictions"])
    main = manifest["main_figure"]
    assert main["exact_match_to_overleaf_project_export"] is True
    assert hashlib.sha256((directory / main["file"]).read_bytes()).hexdigest() == main["sha256"]


def test_readme_ends_with_framework_and_unlabeled_3x3_gallery():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "<details>" not in text and "<summary>" not in text
    assert text.startswith("# LDM\n\n## Data Preparation\n")
    assert "## Visualizations" not in text and "Case " not in text
    prose, images = text.split("![Manuscript framework](assets/framework.png)", 1)
    assert "## Acknowledgements" in prose
    assert "<table>" not in prose and "<img" not in prose
    assert text.strip().endswith("</table>")
    gallery = images
    table = ET.fromstring(re.search(r"<table>.*?</table>", gallery, re.S).group(0))
    rows = table.findall("tr")
    assert len(rows) == 3
    assert all(len(row.findall("td")) == 3 for row in rows)
    assert not table.findall(".//th")
    expected = [f"assets/qualitative/case_{number:02d}.png" for number in range(1, 10)]
    assert [image.attrib["src"] for image in table.iter("img")] == expected
    assert [link.attrib["href"] for link in table.iter("a")] == expected
    assert all(image.attrib["width"] == "100%" for image in table.iter("img"))
    assert "assets/qualitative/case_10.png" not in gallery
    assert "assets/reference/case06_historical_reference.png" not in text
    assert "existing results from the unified experiment package" in prose
    assert "not reruns of this standalone release" in prose
    assert "differences between figure labels and actual experiment configurations" in prose
    assert "assets/qualitative/manifest.json" in prose
