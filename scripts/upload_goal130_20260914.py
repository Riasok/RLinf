"""Archive only Goal PPO checkpoint130; preserve every local file."""
import json
import os
import time
from pathlib import Path

from dotenv import dotenv_values
from huggingface_hub import HfApi

ROOT = Path('/data/minjaeoh/RLinf')
OUT = ROOT / 'logs/hf-goal130-upload-20260914'
REPO = 'Riasok/pi05-robot-checkpoints-20260912'
DEST = 'libero-goal-ppo/global_step_130'
SOURCE = ROOT / ('logs/20260912-libero-goal-pi05-ppo-paper-gpu03/train/'
                 'pi05_ppo_libero_goal_pirl_paper_seed42/checkpoints/global_step_130')


def status(state, **extra):
    OUT.mkdir(exist_ok=True)
    data = dict(state=state, pid=os.getpid(), updated=time.time(), repo=REPO,
                destination=DEST, **extra)
    temp = OUT / 'status.tmp'
    temp.write_text(json.dumps(data, indent=2))
    temp.replace(OUT / 'status.json')
    print(json.dumps(data), flush=True)


def main():
    status('preflight')
    api = HfApi(token=dotenv_values('/data/minjaeoh/.env').get('HF_TOKEN'))
    assert api.whoami()['name'] == 'Riasok'
    assert not api.repo_info(REPO).private
    assert (SOURCE / 'actor/model_state_dict/full_weights.pt').is_file()
    assert (SOURCE / 'actor/dcp_checkpoint/.metadata').is_file()
    assert len(list((SOURCE / 'actor/dcp_checkpoint').glob('*.distcp'))) == 4
    records = {str(p.relative_to(SOURCE)): [p.stat().st_size, p.stat().st_mtime_ns]
               for p in SOURCE.rglob('*') if p.is_file()}
    assert all(size > 0 for size, _ in records.values())
    for attempt in range(1, 6):
        try:
            status('uploading', attempt=attempt,
                   total_bytes=sum(size for size, _ in records.values()))
            api.upload_folder(repo_id=REPO, folder_path=SOURCE, path_in_repo=DEST,
                              commit_message='Add Goal PPO checkpoint 130 with resume state')
            remote = {p.path: p.size for p in api.list_repo_tree(
                REPO, path_in_repo=DEST, recursive=True) if hasattr(p, 'size')}
            for name, (size, mtime) in records.items():
                local = (SOURCE / name).stat()
                assert (local.st_size, local.st_mtime_ns) == (size, mtime)
                assert remote.get(f'{DEST}/{name}') == size, name
            manifest = json.dumps(dict(destination=DEST, files=records), indent=2).encode()
            api.upload_file(repo_id=REPO, path_in_repo='manifests/goal_step130.json',
                            path_or_fileobj=manifest)
            status('complete', verified_files=len(records))
            return
        except Exception as exc:
            response = getattr(exc, 'response', None)
            if attempt == 5 or (response is not None and response.status_code in (401, 403)):
                raise
            status('retrying', attempt=attempt, error=type(exc).__name__)
            time.sleep(30)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        status('failed', error=f'{type(exc).__name__}: {exc}')
        raise
