"""Create a separate VIA page with three yawning labels preloaded on startup."""

import argparse
import hashlib
import json
from pathlib import Path

from scripts.prepare_annotation_starter import sha256, via_template

BOOT = "var via = new _via(via_container);"
START = "/* DRIVER_YAWN_PRESET_START */"
END = "/* DRIVER_YAWN_PRESET_END */"


def build(project, output):
    project, output = Path(project).resolve(), Path(output).resolve()
    record = output.with_suffix(".manifest.json")
    if output.exists() or record.exists():
        raise ValueError("Preserving existing labeler; choose a new output name")
    source = project / ".cache/annotation_tools/via_video_annotator.html"
    html = source.read_text(encoding="utf-8")
    if html.count(BOOT) != 1:
        raise ValueError("Unrecognized VIA startup; refusing to alter this version")
    template = via_template({"yawn_yes": "", "yawn_no": "", "yawn_uncertain": ""})
    template["project"]["pname"] = "Yawning labels loaded - add your video with +"
    encoded = json.dumps(template).replace("<", "\\u003c")
    injection = (
        f"\n{START}\nvar driver_yawn_preset = {encoded};\n"
        "via.d.project_load_json(driver_yawn_preset).catch(function(error) {\n"
        "  console.error('Could not load yawning labels', error);\n"
        "});\n" + END
    )
    result = html.replace(BOOT, BOOT + injection)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result, encoding="utf-8")
    metadata = {
        "original_via_sha256": sha256(source),
        "preloaded_page_sha256": sha256(output),
        "preset_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
        "preset_labels": list(template["attribute"]["1"]["options"].values()),
        "annotations_created": 0,
        "original_app_modified": False,
    }
    record.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output", type=Path, default=Path("data/annotations/starter/yawn_labeler.html")
    )
    args = parser.parse_args()
    print(json.dumps(build(args.project, args.output), indent=2))


if __name__ == "__main__":
    main()
