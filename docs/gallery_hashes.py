# /// script
# dependencies = [
#   "imagehash",
#   "Pillow",
#   "tqdm"
# ]
# ///

import argparse
import json
import sys
from pathlib import Path

import imagehash
from PIL import Image
from tqdm import tqdm


def hash_file(filepath: Path, hash_size: int = 64):
    "Generate a perceptual hash of a figure."
    # The higher the hash_size, the more detail can be captured.
    return imagehash.phash(Image.open(filepath), hash_size=hash_size)


def find_files(path: Path, pattern: str, exclude: list[str]) -> list[Path]:
    "Find all files matching the pattern but not the exclude pattern."
    matches = set(path.rglob(pattern))
    for entry in exclude:
        matches -= set(path.rglob(entry))
    return sorted(matches)


def write_hashes(
    docs_dir: Path, pattern: str, exclude: list[str], hashes_file: Path
) -> None:
    "Write hashes for gallery examples figures to a json file."
    fig_paths = find_files(docs_dir, pattern, exclude)
    hashes = {str(file): str(hash_file(file)) for file in tqdm(fig_paths)}

    with Path.open(hashes_file, "w") as f:
        json.dump(hashes, f, indent=4, sort_keys=True)


def compare_hashes(
    docs_dir: Path,
    pattern: str,
    exclude: list[str],
    hashes_file: Path,
    threshold: int = 10,
) -> None:
    "Check gallery example figure hashes match the stored reference hashes."
    fig_paths = find_files(docs_dir, pattern, exclude)
    hashes = {str(file): hash_file(file) for file in tqdm(fig_paths)}

    msg = ""

    with Path.open(hashes_file) as json_file:
        ref = {
            k: imagehash.hex_to_hash(v) for k, v in json.load(json_file).items()
        }
        common_keys = set(ref.keys()).intersection(hashes.keys())
        diffs = {key: hashes[key] - ref[key] for key in common_keys}
        if not all(diff < threshold for diff in diffs.values()):
            failcases = sorted(
                [f"{k}: delta={v}" for k, v in diffs.items() if v < threshold]
            )
            msg += (
                "Some figure hashes for the gallery have changed.\n"
                "Please check, whether the following figures look as expected:"
                "\n\n" + "\n".join(failcases) + "\n\n"
            )
        if len(ref) < len(hashes):
            delta = sorted(set(hashes.keys()) - set(ref.keys()))
            msg += (
                "For the following figures there is no stored hash:"
                "\n\n" + "\n".join(delta) + "\n\n"
            )
        if len(ref) > len(hashes):
            delta = sorted(set(ref.keys()) - set(hashes.keys()))
            msg += (
                "For the following figures there is a stored hash, but they "
                "are not generated anymore.\nIf this is intentional, "
                f"please remove them from {hashes_file}:"
                "\n\n" + "\n".join(delta) + "\n\n"
            )
    if msg != "":
        msg += (
            "If all figures look good, please run `make gallery_hashes` "
            "to update the stored hashes."
        )
        print("#" * 80)
        print(msg)
        print("#" * 80)
        sys.exit(123)
    print("Gallery figure hashes are equal to references hashes.")


def main():
    parser = argparse.ArgumentParser(description="Process gallery hashes.")
    parser.add_argument(
        "action",
        type=str,
        choices=["compare", "write"],
        help="Action to perform: 'compare' or 'write'.",
    )
    parser.add_argument(
        "--docs_dir",
        type=str,
        default="docs/_build/html/_images",
        help="Directory containing documents. Default: 'docs'.",
    )
    parser.add_argument(
        "--hashes_file",
        type=str,
        default="docs/gallery_hashes.json",
        help="File containing the hashes. Default: 'docs/gallery_hashes.json'.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="sphx_glr_plot_*.png",
        help="Pattern for document filenames. Default: 'sphx_glr_plot_*.png'.",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        nargs="+",
        default=[],
        help="Pattern for excluded document filenames. Default: []",
    )

    args = parser.parse_args()

    if args.action == "compare":
        compare_hashes(
            Path(args.docs_dir),
            args.pattern,
            args.exclude,
            Path(args.hashes_file),
        )
    elif args.action == "write":
        write_hashes(
            Path(args.docs_dir),
            args.pattern,
            args.exclude,
            Path(args.hashes_file),
        )
    else:
        print("Invalid action. Use 'compare' or 'write'.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
