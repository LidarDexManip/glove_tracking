import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download one HOT3D clip tar from Hugging Face."
    )
    parser.add_argument("--repo-id", default="bop-benchmark/hot3d")
    parser.add_argument("--subset", default="train_quest3")
    parser.add_argument("--clip-name", default="clip-000000.tar")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("hot3d_data") / "train_quest3",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if the local output file already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    local_path = args.output_dir / args.clip_name

    if local_path.exists() and not args.force:
        print(f"Already exists: {local_path.resolve()}")
        return

    downloaded = hf_hub_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        filename=f"{args.subset}/{args.clip_name}",
        force_download=args.force,
    )
    shutil.copy2(downloaded, local_path)
    print(f"Downloaded to: {local_path.resolve()}")


if __name__ == "__main__":
    main()
