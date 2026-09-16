"""Allow current Spatial evaluator to finish before restarting updated queue."""
import json
import signal
import subprocess
import time
from pathlib import Path

import psutil

ROOT = Path('/data/minjaeoh/RLinf')
OUT = ROOT/'logs/20260914-crosssuite-adarms-lora'


def main():
    controller = psutil.Process(4066952)
    evaluator = psutil.Process(166293)
    assert 'scripts/queue_crosssuite_ablation_20260914.py' in controller.cmdline()
    assert evaluator.ppid() == controller.pid
    assert json.loads((OUT/'status.json').read_text())['job'] == 'lora80_libero_spatial'
    controller.send_signal(signal.SIGSTOP)
    print('Queue controller paused; active Spatial evaluator continues.',flush=True)
    while evaluator.is_running() and evaluator.status() != psutil.STATUS_ZOMBIE:
        time.sleep(15)
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    dest = OUT/'lora80_libero_spatial'
    events = EventAccumulator(str(dest/'tensorboard')); events.Reload()
    metrics = {tag:events.Scalars(tag)[-1].value for tag in events.Tags()['scalars']
               if tag.startswith('eval/')}
    assert metrics.get('eval/num_trajectories') == 500, metrics
    assert 'eval/success_once' in metrics
    import queue_crosssuite_ablation_20260914 as queue
    result = dict(name=dest.name,suite='libero_spatial',k=5,num_steps=3,
        checkpoint=str(queue.LORA_ROOT/'global_step_80/actor/model_state_dict/full_weights.pt'),
        model_path=str(ROOT/'checkpoints/RLinf-Pi05-LIBERO-SFT'),model_config='pi05_libero',
        benchmark_mode='standard',metrics=metrics)
    queue.write(dest/'result.json',result)
    queue.summarize(dest.name,'libero_spatial',result)
    controller.terminate()
    controller.send_signal(signal.SIGCONT)
    psutil.wait_procs([controller],timeout=30)
    print('Spatial result preserved; restarting queue with suite-specific denoising.',flush=True)
    subprocess.run([str(ROOT/'.venv/bin/python'),'-u',
                    'scripts/queue_crosssuite_ablation_20260914.py'],cwd=ROOT,check=True)


if __name__ == '__main__':
    main()
