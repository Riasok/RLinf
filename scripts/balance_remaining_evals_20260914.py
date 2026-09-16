"""Finish current GPU3 evaluation, then share remaining jobs across GPUs0-3."""
import concurrent.futures
import json
import queue
import signal
import subprocess
import time

import psutil

import queue_crosssuite_ablation_20260914 as helpers
import queue_goal130_ns3_20260914 as reruns
import run_csd_diagnosis as diag
import run_pirl130_20260906 as runtime

OUT = helpers.OUT
CONTROL = OUT/'balanced_queue'


def record(state, **extra):
    helpers.write(CONTROL/'status.json',dict(state=state,time=time.strftime('%F %T %Z'),**extra))
    print(state,extra,flush=True)


def wait_gpu(gpu):
    while True:
        if gpu < 3:
            p=reruns.OUT/f'gpu{gpu}_status.json'
            try: ready=json.loads(p.read_text()).get('state')=='complete'
            except (FileNotFoundError,json.JSONDecodeError): ready=False
            if not ready:
                time.sleep(20)
                continue
        rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used',
                                      '--format=csv,noheader,nounits'],text=True)
        busy=any(int(r.split(',')[0])==gpu and int(r.split(',')[1])>1000 for r in rows.splitlines())
        if not busy:return
        time.sleep(20)


def lane(gpu, jobs):
    while True:
        if jobs.empty(): break
        helpers.write(CONTROL/f'gpu{gpu}.json',dict(state='waiting_for_lane_release',gpu=gpu))
        wait_gpu(gpu)
        try:name,suite,checkpoint,ns,lora=jobs.get_nowait()
        except queue.Empty:break
        helpers.write(CONTROL/f'gpu{gpu}.json',dict(state='evaluating',job=name,gpu=gpu))
        overrides=['actor.seed=42','+rollout.model.openpi.action_horizon=10']
        if checkpoint:
            overrides+=['runner.ckpt_strict=true','rollout.model.add_value_head=true',
                        'rollout.model.openpi.add_value_head=true','rollout.model.openpi.value_after_vlm=true']
        if lora:
            overrides+=['rollout.model.is_lora=true','rollout.model.lora_rank=32',
                        '+rollout.model.lora_target=action_expert']
        result=diag.evaluate((name,suite,5,ns,checkpoint),str(gpu),trials=50,envs=20,
            epochs=25,expected=500,timeout=21600,extra_overrides=overrides)
        helpers.summarize(name,suite,result)
        jobs.task_done()
    helpers.write(CONTROL/f'gpu{gpu}.json',dict(state='complete',gpu=gpu))


def main():
    controller=psutil.Process(222515)
    evaluator=psutil.Process(281281)
    assert 'scripts/queue_crosssuite_ablation_20260914.py' in controller.cmdline()
    assert evaluator.ppid()==controller.pid
    assert json.loads((OUT/'status.json').read_text())['job']=='lora80_libero_goal'
    controller.send_signal(signal.SIGSTOP)
    record('waiting_for_active_lora_goal_to_finish',controller_pid=controller.pid)
    while evaluator.is_running() and evaluator.status()!=psutil.STATUS_ZOMBIE:
        time.sleep(15)
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    dest=OUT/'lora80_libero_goal'
    events=EventAccumulator(str(dest/'tensorboard'));events.Reload()
    metrics={tag:events.Scalars(tag)[-1].value for tag in events.Tags()['scalars'] if tag.startswith('eval/')}
    assert metrics.get('eval/num_trajectories')==500,metrics
    assert 'eval/success_once' in metrics
    result=dict(name=dest.name,suite='libero_goal',k=5,num_steps=5,
        checkpoint=str(helpers.LORA_ROOT/'global_step_80/actor/model_state_dict/full_weights.pt'),
        model_path=str(runtime.ROOT/'checkpoints/RLinf-Pi05-LIBERO-SFT'),
        model_config='pi05_libero',benchmark_mode='standard',metrics=metrics)
    helpers.write(dest/'result.json',result)
    helpers.summarize(dest.name,'libero_goal',result)
    controller.terminate();controller.send_signal(signal.SIGCONT)
    psutil.wait_procs([controller],timeout=30)
    runtime.setup_environment();diag.OUT=OUT
    if not (OUT/'audit_adarms_only.json').exists():
        helpers.audit_checkpoint(helpers.ADA,'adarms_only')
    plan=json.loads((OUT/'plan.json').read_text())
    remaining=[job for job in plan['jobs'] if not (OUT/job[0]/'result.json').exists()]
    helpers.write(CONTROL/'plan.json',dict(jobs=remaining,gpus=[0,1,2,3],
        prerequisite_for_012=str(reruns.OUT),protocol='unchanged single-GPU rank0 seed42'))
    work=queue.Queue()
    for job in remaining:work.put(job)
    helpers.status('evaluating_parallel',gpus=[0,1,2,3],remaining=len(remaining))
    record('balancing_remaining_jobs',remaining=len(remaining))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(lane,gpu,work) for gpu in range(4)]
        for future in concurrent.futures.as_completed(futures):future.result()
    helpers.write(OUT/'results.json',{p.parent.name:json.loads(p.read_text()) for p in OUT.glob('*/per_task.json')})
    helpers.status('complete',total=len(plan['jobs']))
    record('complete')


if __name__=='__main__':
    try:main()
    except Exception as exc:
        record('failed',error=f'{type(exc).__name__}: {exc}')
        raise
