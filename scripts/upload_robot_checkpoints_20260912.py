"""Upload a frozen inventory of requested robot checkpoints; never delete locally."""
import json
import os
import time
from pathlib import Path

from dotenv import dotenv_values
from huggingface_hub import HfApi

ROOT = Path('/data/minjaeoh/RLinf')
JOB = ROOT / 'logs/hf-checkpoint-upload-public-20260913'
REPO = 'Riasok/pi05-robot-checkpoints-20260912'
FAMILIES = {
    'libero-goal-ppo': '20260912-libero-goal-pi05-ppo-paper-gpu03',
    'spatial73-ppo': '20260907-spatial73',
    'spatial73-bc': '20260908-spatial73-bc',
    'spatial10-adarms-only-ppo': '20260910-spatial10-ppo-adarms-only',
    'spatial73-attention-ffn-lora-ppo': '20260910-spatial73-ppo-action-lora-r32',
}


def save_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def main():
    JOB.mkdir(exist_ok=True)
    token = dotenv_values('/data/minjaeoh/.env').get('HF_TOKEN')
    api = HfApi(token=token)
    assert api.whoami()['name'] == 'Riasok'
    manifest_path = JOB / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = []
        for family, directory in FAMILIES.items():
            paths = sorted((ROOT / 'logs' / directory).rglob('global_step_*'),
                           key=lambda p: int(p.name.rsplit('_', 1)[1]))
            for path in paths:
                if not path.is_dir():
                    continue
                files = list(path.rglob('*'))
                assert (path / 'actor/model_state_dict/full_weights.pt').is_file(), path
                assert (path / 'actor/dcp_checkpoint/.metadata').is_file(), path
                assert len(list((path / 'actor/dcp_checkpoint').glob('*.distcp'))) == 4, path
                records = {str(p.relative_to(path)): [p.stat().st_size, p.stat().st_mtime_ns]
                           for p in files if p.is_file()}
                assert all(size > 0 for size, _ in records.values())
                manifest.append({'source': str(path.relative_to(ROOT)),
                                 'destination': f'{family}/{path.name}', 'files': records})
        save_json(manifest_path, manifest)
    api.create_repo(repo_id=REPO, private=False, exist_ok=True)
    api.update_repo_settings(repo_id=REPO, private=False)
    assert not api.repo_info(REPO).private
    readme = ('---\nlicense: other\nlicense_name: upstream-licenses-apply\n'
              'license_link: https://github.com/RLinf/RLinf\n---\n'
              '# Robot checkpoint archive\n\n'
              'Public research archive of five pi0.5 experiment families. '
              'PPO+CSD is intentionally excluded. Checkpoint names denote local run iterations; '
              'BC iteration counts are not necessarily optimizer-step-equivalent to PPO.\n\n'
              'Each checkpoint contains `actor/model_state_dict/full_weights.pt` for the '
              'RLinf model loader and distributed `actor/dcp_checkpoint` training state. '
              'These are not standalone Transformers or LeRobot exports. LoRA checkpoints '
              'include the full model state, not adapter-only exports. Loading requires the '
              'matching RLinf/OpenPI code, configuration, base assets and normalization statistics. '
              'Resume support depends on the runner; these files do not imply exact environment/RNG replay.\n\n'
              'See manifest.json for the frozen checkpoint inventory. Local files are not deleted.\n')
    api.upload_file(repo_id=REPO, path_or_fileobj=readme.encode(), path_in_repo='README.md')
    api.upload_file(repo_id=REPO, path_or_fileobj=str(manifest_path), path_in_repo='manifest.json')
    configs = [
        'libero_goal_ppo_openpi_pi05_gpu03_20260912.yaml',
        'libero_spatial10_ppo_adarms_only_20260910.yaml',
        'libero_spatial73_ppo_lora_20260910.yaml',
        'libero_spatial_7train3heldout_pi05_20260907.yaml',
    ]
    for name in configs:
        api.upload_file(repo_id=REPO, path_or_fileobj=str(ROOT / 'examples/embodiment/config' / name),
                        path_in_repo='configs/' + name)
    status_path = JOB / 'status.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {'completed': []}
    status.update(repo=REPO, total=len(manifest), total_bytes=sum(
        size for item in manifest for size, _ in item['files'].values()), pid=os.getpid())
    for item in manifest:
        dest = item['destination']
        if dest in status['completed']:
            continue
        source = ROOT / item['source']
        for name, (size, mtime) in item['files'].items():
            stat = (source / name).stat()
            assert (stat.st_size, stat.st_mtime_ns) == (size, mtime), f'Changed checkpoint: {source / name}'
        status.update(state='uploading', current=dest, updated=time.time())
        save_json(status_path, status)
        print('Uploading', dest, flush=True)
        for attempt in range(5):
            try:
                api.upload_folder(repo_id=REPO, folder_path=source, path_in_repo=dest,
                                  commit_message=f'Archive {dest}')
                remote = {entry.path: entry.size for entry in api.list_repo_tree(
                    REPO, path_in_repo=dest, recursive=True) if hasattr(entry, 'size')}
                for name, (size, _) in item['files'].items():
                    assert remote.get(f'{dest}/{name}') == size, f'Remote size mismatch: {dest}/{name}'
                break
            except Exception as exc:
                status.update(state='retrying', error=type(exc).__name__, updated=time.time())
                response = getattr(exc, 'response', None)
                if response is not None and response.status_code == 403:
                    status.update(state='blocked', error='HTTP 403')
                    save_json(status_path, status)
                    raise
                save_json(status_path, status)
                if attempt == 4:
                    status['state'] = 'failed'
                    save_json(status_path, status)
                    raise
                time.sleep(30)
        status['completed'].append(dest)
        status.update(state='verified', updated=time.time())
        save_json(status_path, status)
        print('Verified', dest, flush=True)
    status.update(state='complete', current=None, updated=time.time())
    save_json(status_path, status)


if __name__ == '__main__':
    main()
