# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python
"""Upload stable GR00T checkpoints every N steps and verify remote file sizes."""

import argparse
import json
import os
import time
import traceback
from pathlib import Path


def load_dotenv(path: str) -> None:
    for raw_line in Path(path).read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %Z")


class Gr00tCheckpointWatcher:
    def __init__(self, args: argparse.Namespace) -> None:
        from huggingface_hub import HfApi

        self.args = args
        self.api = HfApi(token=os.environ["HF_TOKEN"])
        self.roots = [Path(path) for path in args.checkpoint_dir]
        self.state_dir = Path(args.state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.state_dir / "watcher.log"
        self.uploaded_path = self.state_dir / "uploaded.json"
        self.uploaded = (
            json.loads(self.uploaded_path.read_text())
            if self.uploaded_path.exists()
            else {}
        )
        self.observed: dict[str, dict] = {}
        self.retry_after: dict[str, float] = {}

    def log(self, message: str) -> None:
        with self.log_path.open("a") as handle:
            handle.write(f"{now()} {message}\n")

    def write_json(self, path: Path, payload: dict) -> None:
        temporary = path.with_suffix(path.suffix + ".temporary")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
        temporary.replace(path)

    def status(self, state: str, **extra) -> None:
        self.write_json(
            self.state_dir / "status.json",
            {"time": now(), "pid": os.getpid(), "state": state, **extra},
        )

    def checkpoint_snapshot(self, directory: Path) -> dict | None:
        metadata = directory / "actor/dcp_checkpoint/.metadata"
        weights = directory / "actor/model_state_dict/full_weights.pt"
        shards = sorted((directory / "actor/dcp_checkpoint").glob("*.distcp"))
        if not metadata.is_file() or not weights.is_file() or len(shards) != 4:
            return None
        files = [path for path in directory.rglob("*") if path.is_file()]
        if any(path.name.endswith(".temporary") for path in files):
            return None
        cutoff = time.time() - self.args.min_age_seconds
        if any(path.stat().st_size <= 0 or path.stat().st_mtime > cutoff for path in files):
            return None
        return {
            str(path.relative_to(directory)): [path.stat().st_size, path.stat().st_mtime_ns]
            for path in files
        }

    def find_checkpoint(self, step: int) -> tuple[Path, dict] | None:
        for root in self.roots:
            directory = root / f"global_step_{step}"
            snapshot = self.checkpoint_snapshot(directory)
            if snapshot is not None:
                return directory, snapshot
        return None

    def remote_sizes(self, destination: str) -> dict[str, int]:
        try:
            entries = self.api.list_repo_tree(
                self.args.repo, path_in_repo=destination, recursive=True
            )
            prefix = destination + "/"
            return {
                entry.path[len(prefix) :]: entry.size
                for entry in entries
                if getattr(entry, "size", None) is not None
                and entry.path.startswith(prefix)
            }
        except Exception:
            return {}

    def remote_verified(self, snapshot: dict, destination: str) -> bool:
        remote = self.remote_sizes(destination)
        return bool(snapshot) and all(
            remote.get(relative_path) == values[0]
            for relative_path, values in snapshot.items()
        )

    def save_uploaded(self) -> None:
        self.write_json(self.uploaded_path, self.uploaded)

    def upload_metadata(self) -> None:
        for source in self.args.metadata_file:
            path = Path(source)
            if not path.is_file():
                continue
            self.api.upload_file(
                repo_id=self.args.repo,
                path_or_fileobj=str(path),
                path_in_repo=f"run_metadata/{path.name}",
            )

    def upload_checkpoint(self, step: int, directory: Path, snapshot: dict) -> None:
        name = f"global_step_{step}"
        if self.remote_verified(snapshot, name):
            self.log(f"{name}: already verified on Hub")
        else:
            total_bytes = sum(values[0] for values in snapshot.values())
            self.log(f"{name}: uploading {total_bytes / 1e9:.2f} GB from {directory}")
            self.status("uploading", step=step, uploaded=sorted(self.uploaded))
            self.api.upload_folder(
                repo_id=self.args.repo,
                folder_path=str(directory),
                path_in_repo=name,
                commit_message=f"Auto-upload GR00T checkpoint step {step}",
                ignore_patterns=["*.temporary", ".*.temporary"],
            )
            current = self.checkpoint_snapshot(directory)
            if current != snapshot:
                raise RuntimeError("checkpoint changed during upload")
            if not self.remote_verified(snapshot, name):
                raise RuntimeError("remote file-size verification failed")
            self.log(f"{name}: upload complete and size-verified")
        self.uploaded[name] = {
            "verified": now(),
            "source": str(directory),
            "bytes": sum(values[0] for values in snapshot.values()),
        }
        self.save_uploaded()

    def run(self) -> int:
        self.api.create_repo(
            self.args.repo,
            repo_type="model",
            private=not self.args.public,
            exist_ok=True,
        )
        self.upload_metadata()
        self.log(
            f"WATCHER START repo={self.args.repo} roots="
            + ",".join(str(root) for root in self.roots)
        )
        expected_steps = list(
            range(self.args.interval, self.args.max_step + 1, self.args.interval)
        )
        while True:
            for step in expected_steps:
                name = f"global_step_{step}"
                if name in self.uploaded or self.retry_after.get(name, 0) > time.time():
                    continue
                found = self.find_checkpoint(step)
                if found is None:
                    self.observed.pop(name, None)
                    continue
                directory, snapshot = found
                observation = {"source": str(directory), "snapshot": snapshot}
                if self.observed.get(name) != observation:
                    self.observed[name] = observation
                    self.log(f"{name}: complete candidate observed; waiting one poll")
                    continue
                try:
                    self.upload_checkpoint(step, directory, snapshot)
                    self.retry_after.pop(name, None)
                except Exception as error:
                    self.retry_after[name] = time.time() + self.args.retry_seconds
                    self.log(f"{name}: upload failed: {type(error).__name__}: {error}")
            uploaded_steps = sorted(
                int(name.rsplit("_", 1)[1]) for name in self.uploaded
            )
            if self.args.max_step in uploaded_steps:
                self.upload_metadata()
                self.status("complete", uploaded_steps=uploaded_steps)
                self.log("WATCHER COMPLETE: step 150 verified on Hub")
                return 0
            self.status(
                "watching",
                uploaded_steps=uploaded_steps,
                retry_after=self.retry_after,
                next_poll_seconds=self.args.poll_seconds,
            )
            time.sleep(self.args.poll_seconds)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", action="append", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--metadata-file", action="append", default=[])
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--max-step", type=int, default=150)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--min-age-seconds", type=int, default=60)
    parser.add_argument("--retry-seconds", type=int, default=300)
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--dotenv", default="/data/minjaeoh/.env")
    args = parser.parse_args()
    load_dotenv(args.dotenv)
    if "HF_TOKEN" not in os.environ:
        print("HF_TOKEN missing", file=os.sys.stderr)
        return 2
    try:
        return Gr00tCheckpointWatcher(args).run()
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
