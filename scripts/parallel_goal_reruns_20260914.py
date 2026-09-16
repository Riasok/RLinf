"""Run the six existing suite-corrected reruns on three independent GPUs."""
import concurrent.futures
import json
import subprocess
import time

import queue_crosssuite_ablation_20260914 as helpers
import queue_goal130_ns3_20260914 as reruns
import run_csd_diagnosis as diag
import run_pirl130_20260906 as runtime

OUT = reruns.OUT


def lane(gpu, jobs):
    for name, suite, checkpoint in jobs:
        if not (OUT/name/'result.json').exists():
            while True:
                rows = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used',
                                               '--format=csv,noheader,nounits'],text=True)
                busy = any(int(r.split(',')[0]) == gpu and int(r.split(',')[1]) > 1000
                           for r in rows.splitlines())
                if not busy: break
                helpers.write(OUT/f'gpu{gpu}_status.json',dict(state='waiting_for_gpu',job=name))
                time.sleep(30)
        ns = 3 if suite == 'libero_spatial' else 5
        helpers.write(OUT/f'gpu{gpu}_status.json',dict(state='evaluating',job=name,
            gpu=gpu,denoising_steps=ns,time=time.strftime('%F %T %Z')))
        overrides = ['actor.seed=42','+rollout.model.openpi.action_horizon=10',
                     'runner.ckpt_strict=true','rollout.model.add_value_head=true',
                     'rollout.model.openpi.add_value_head=true',
                     'rollout.model.openpi.value_after_vlm=true']
        result = diag.evaluate((name,suite,5,ns,checkpoint),str(gpu),trials=50,
            envs=20,epochs=25,expected=500,timeout=21600,extra_overrides=overrides)
        helpers.summarize(name,suite,result)
    helpers.write(OUT/f'gpu{gpu}_status.json',dict(state='complete',gpu=gpu))


def main():
    runtime.setup_environment()
    diag.OUT = OUT
    helpers.OUT = OUT
    plan = json.loads((OUT/'plan.json').read_text())
    jobs = plan['jobs']
    assignments = {gpu: [job for job in jobs if job[1] == suite]
                   for gpu,suite in enumerate(['libero_spatial','libero_object','libero_goal'])}
    plan.update(gpu=None,lanes=assignments,prerequisite=None,
                scheduling='parallel independent single-GPU evaluations, same local rank/seed')
    helpers.write(OUT/'plan.json',plan)
    reruns.status('evaluating_parallel',gpus=[0,1,2],total=len(jobs))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(lane,gpu,lane_jobs) for gpu,lane_jobs in assignments.items()]
        for future in concurrent.futures.as_completed(futures): future.result()
    helpers.write(OUT/'results.json',{p.parent.name:json.loads(p.read_text())
                  for p in OUT.glob('*/per_task.json')})
    reruns.status('complete',total=len(jobs))


if __name__ == '__main__':
    try: main()
    except Exception as exc:
        reruns.status('failed',error=f'{type(exc).__name__}: {exc}')
        raise
