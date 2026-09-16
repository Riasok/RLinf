"""Queue suite-specific 3/5/5 Goal130/SFT-AdaRMS evaluations after current jobs."""
import json
import subprocess
import time

import queue_crosssuite_ablation_20260914 as previous
import run_csd_diagnosis as diag
import run_pirl130_20260906 as runtime

PRIOR = previous.OUT
OUT = runtime.ROOT / 'logs/20260914-goal130-ns3-rerun'


def status(state, **extra):
    previous.write(OUT / 'status.json', dict(state=state, time=time.strftime('%F %T %Z'), **extra))
    print(state, extra, flush=True)


def main():
    OUT.mkdir(exist_ok=True)
    jobs = [(f'{label}_ns{3 if suite == "libero_spatial" else 5}_{suite}', suite, str(checkpoint))
            for label, checkpoint in [('goal130', previous.GOAL),
                                      ('goal130_sftadarms', previous.SWAP)]
            for suite in ['libero_spatial', 'libero_object', 'libero_goal']]
    previous.write(OUT / 'plan.json', dict(jobs=jobs, denoising_steps={'spatial':3,'object':5,'goal':5},
        generation=10, execution=5, policy_seed=42, gpu=3, episodes_per_job=500,
        prerequisite=str(PRIOR), note='Only denoising count changes from the matched single-GPU protocol.'))
    while True:
        try:
            prior = json.loads((PRIOR/'status.json').read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            prior = {}
        if prior.get('state') == 'complete':
            break
        if prior.get('state') == 'failed':
            raise RuntimeError('Earlier evaluation queue failed; refusing to race remaining workers')
        status('waiting_for_current_evaluation_queue', prior_job=prior.get('job'),
               prior_index=prior.get('job_index'))
        time.sleep(30)
    runtime.setup_environment()
    diag.OUT = OUT
    previous.OUT = OUT
    for index, (name, suite, checkpoint) in enumerate(jobs):
        while True:
            rows = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used',
                                           '--format=csv,noheader,nounits'],text=True)
            busy = any(int(r.split(',')[0]) == 3 and int(r.split(',')[1]) > 1000
                       for r in rows.splitlines())
            if not busy: break
            status('waiting_for_gpu3', job=name)
            time.sleep(30)
        overrides = ['actor.seed=42', '+rollout.model.openpi.action_horizon=10',
                     'runner.ckpt_strict=true', 'rollout.model.add_value_head=true',
                     'rollout.model.openpi.add_value_head=true',
                     'rollout.model.openpi.value_after_vlm=true']
        status('evaluating', job=name, job_index=index+1, total=len(jobs))
        ns = 3 if suite == 'libero_spatial' else 5
        result = diag.evaluate((name,suite,5,ns,checkpoint),'3',trials=50,envs=20,
            epochs=25,expected=500,timeout=21600,extra_overrides=overrides)
        previous.summarize(name,suite,result)
        previous.write(OUT/'results.json', {p.parent.name:json.loads(p.read_text())
                       for p in OUT.glob('*/per_task.json')})
    status('complete', total=len(jobs))


if __name__ == '__main__':
    try: main()
    except Exception as exc:
        status('failed', error=f'{type(exc).__name__}: {exc}')
        raise
