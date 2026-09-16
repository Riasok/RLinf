"""Standalone Goal-130 evaluation, separate from the interrupted queue."""
import json
import os
import subprocess
import time

import run_csd_diagnosis as diag
import run_pirl130_20260906 as runtime

OUT = runtime.ROOT / 'logs/20260914-goal130-eval-retry'
CHECKPOINT = runtime.ROOT / (
    'logs/20260912-libero-goal-pi05-ppo-paper-gpu03/train/'
    'pi05_ppo_libero_goal_pirl_paper_seed42/checkpoints/'
    'global_step_130/actor/model_state_dict/full_weights.pt'
)


def status(state, **extra):
    record = dict(state=state, pid=os.getpid(), time=time.strftime('%F %T %Z'),
                  checkpoint=str(CHECKPOINT), gpus=[0, 1, 2, 3], **extra)
    path = OUT / 'status.json'
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(record, indent=2))
    temp.replace(path)
    print(json.dumps(record), flush=True)


def main():
    OUT.mkdir(exist_ok=False)
    try:
        assert CHECKPOINT.is_file()
        rows = subprocess.check_output([
            'nvidia-smi', '--query-gpu=index,memory.used',
            '--format=csv,noheader,nounits'], text=True)
        for row in rows.splitlines():
            gpu, memory = map(int, row.split(','))
            if gpu < 4 and memory > 1000:
                raise RuntimeError(f'GPU {gpu} is occupied; refusing to launch')
        runtime.setup_environment()
        diag.OUT = OUT
        status('evaluating', episodes=500, generation=10, execution=5, denoising_steps=5,
               seed=42, episode_horizon=320, parallel_envs=20)
        result = diag.evaluate(
            ('goal_step130', 'libero_goal', 5, 5, str(CHECKPOINT)),
            '0-3', trials=50, envs=20, epochs=25, expected=500, timeout=21600)
        status('complete', result=result)
    except Exception as exc:
        status('failed', error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    main()
