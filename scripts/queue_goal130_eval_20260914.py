"""Stop only the owned Goal run after save130, then evaluate its final policy."""
import argparse
import json
import re
import time
from pathlib import Path

import psutil
import subprocess

import run_pirl130_20260906 as runtime

ROOT = runtime.ROOT
OUT = ROOT / 'logs/20260914-goal130-stop-eval'
TRAIN = ROOT / 'logs/20260912-libero-goal-pi05-ppo-paper-gpu03/train'
CKPT = TRAIN / 'pi05_ppo_libero_goal_pirl_paper_seed42/checkpoints/global_step_130'
SUPERVISOR = 3381297


def status(state, **kwargs):
    OUT.mkdir(exist_ok=True)
    target = OUT / 'status.json'
    tmp = target.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(state=state, time=time.strftime('%F %T %Z'),
                                  queue_pid=psutil.Process().pid, **kwargs), indent=2))
    tmp.replace(target)
    print(state, kwargs, flush=True)


def checkpoint_snapshot():
    weights = CKPT / 'actor/model_state_dict/full_weights.pt'
    metadata = CKPT / 'actor/dcp_checkpoint/.metadata'
    shards = list((CKPT / 'actor/dcp_checkpoint').glob('*.distcp'))
    if not weights.is_file() or not metadata.is_file() or len(shards) != 4:
        return None
    files = [weights, metadata, *sorted(shards)]
    result = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in files]
    return result if all(size > 0 for _, size, _ in result) else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    supervisor = psutil.Process(SUPERVISOR)
    assert 'scripts/run_libero_goal_pi05_ppo_paper_20260912.py' in supervisor.cmdline()
    assert psutil.Process(3381330).ppid() == SUPERVISOR
    print('Validated supervisor and trainer. Plan: save130 -> scoped stop -> Goal500 eval. '
          'Next PPO waits for task-suite confirmation.', flush=True)
    if not args.launch:
        return
    runtime.setup_environment()
    previous = None
    stable_since = None
    while True:
        if not supervisor.is_running() or supervisor.status() == psutil.STATUS_ZOMBIE:
            raise RuntimeError('Goal supervisor exited before the scheduled stop')
        text = (TRAIN / 'metrics.log').read_text()
        steps = [int(x) for x in re.findall(r'Global Step:\s*(\d+)', text)]
        latest = max(steps, default=0)
        current = checkpoint_snapshot()
        if current and current == previous and latest >= 130:
            if stable_since is None:
                stable_since = time.monotonic()
            if time.monotonic() - stable_since >= 45:
                break
        else:
            stable_since = None
        previous = current
        status('waiting_for_complete_checkpoint_130', latest_step=latest)
        time.sleep(15)
    status('stopping_goal_supervisor', checkpoint=str(CKPT))
    # Its registered SIGTERM handler cleans up only its tracked process tree.
    supervisor.terminate()
    deadline = time.monotonic() + 900
    while supervisor.is_running() and supervisor.status() != psutil.STATUS_ZOMBIE:
        if time.monotonic() > deadline:
            raise RuntimeError('Supervisor cleanup timeout; refusing to launch evaluation')
        time.sleep(10)
    deadline = time.monotonic() + 900
    while True:
        rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used',
                                       '--format=csv,noheader,nounits'], text=True)
        busy = [line for line in rows.splitlines()
                if int(line.split(',')[0]) < 4 and int(line.split(',')[1]) > 1000]
        if not busy:
            break
        if time.monotonic() > deadline:
            raise RuntimeError('GPUs 0-3 remain occupied; refusing to interfere')
        status('waiting_for_gpu_release', busy=busy)
        time.sleep(15)
    import run_csd_diagnosis as diag
    diag.OUT = OUT / 'eval'
    status('evaluating_goal_step130', episodes=500, gpus=[0, 1, 2, 3])
    result = diag.evaluate(('goal_step130', 'libero_goal', 5, 5,
                            str(CKPT / 'actor/model_state_dict/full_weights.pt')),
                           '0-3', trials=50, envs=20, epochs=25, expected=500,
                           timeout=21600)
    status('evaluation_complete_next_training_requires_suite_confirmation', result=result)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        status('failed', error=f'{type(exc).__name__}: {exc}')
        raise
