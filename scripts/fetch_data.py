#!/usr/bin/env python3
"""Download a named, immutable Hugging Face dataset revision."""
import argparse
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-id', required=True)
    parser.add_argument('--revision', required=True, help='Full dataset commit SHA')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'[0-9a-f]{40}', args.revision):
        parser.error('--revision must be an immutable 40-character commit SHA')
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=args.repo_id, repo_type='dataset', revision=args.revision,
                      local_dir=str(args.output.expanduser().resolve()))
    from embodied_harness.release import check_dataset
    result = check_dataset(args.output.resolve(), hashes=True)
    if not result['ok']:
        raise SystemExit('Dataset verification failed: ' + '; '.join(result['errors']))
    print(f"Verified {result['cases']} cases in {args.output.resolve()}")


if __name__ == '__main__':
    main()
