"""Serial, reproducible cross-suite evaluations and AdaRMS weight-swap ablation."""
import json
import re
import time
import subprocess
from pathlib import Path

import run_csd_diagnosis as diag
import run_pirl130_20260906 as runtime

ROOT = runtime.ROOT
OUT = ROOT / 'logs/20260914-crosssuite-adarms-lora'
BASE = ROOT / 'checkpoints/RLinf-Pi05-LIBERO-SFT/model.safetensors'
GOAL = ROOT / ('logs/20260912-libero-goal-pi05-ppo-paper-gpu03/train/'
    'pi05_ppo_libero_goal_pirl_paper_seed42/checkpoints/global_step_130/actor/model_state_dict/full_weights.pt')
LORA_ROOT = ROOT / ('logs/20260910-spatial73-ppo-action-lora-r32/train/'
    'pi05_ppo_action_lora_r32_spatial7_heldout3_seed42/checkpoints')
ADA = ROOT / ('logs/20260910-spatial10-ppo-adarms-only/train/'
    'pi05_ppo_spatial10_adarms_only_seed42/checkpoints/global_step_40/actor/model_state_dict/full_weights.pt')
PPO = ROOT / ('logs/20260907-spatial73/train/pi05_ppo_spatial7_heldout3_seed42/'
    'checkpoints/global_step_50/actor/model_state_dict/full_weights.pt')
SWAP = OUT / 'derived/goal130_sft_adarms.pt'


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def status(state, **extra):
    write(OUT / 'status.json', dict(state=state, time=time.strftime('%F %T %Z'), **extra))
    print(state, extra, flush=True)


def prepare_swap():
    import torch
    from safetensors import safe_open
    torch.set_num_threads(2)
    original = torch.load(GOAL, map_location='cpu', mmap=True, weights_only=True)
    names = sorted(n for n in original if re.fullmatch(
        r'paligemma_with_expert\.gemma_expert\.model\.(?:layers\.\d+\.(?:input_layernorm|post_attention_layernorm)|norm)\.dense\.(?:weight|bias)', n))
    assert len(names) == 74
    swapped = dict(original)
    with safe_open(BASE, framework='pt', device='cpu') as f:
        for name in names:
            base = f.get_tensor(name)
            assert base.shape == original[name].shape
            swapped[name] = base.to(original[name].dtype).clone()
    SWAP.parent.mkdir(exist_ok=True)
    temp = SWAP.with_suffix('.tmp')
    torch.save(swapped, temp)
    temp.replace(SWAP)
    check = torch.load(SWAP, map_location='cpu', mmap=True, weights_only=True)
    for name in original:
        assert torch.equal(check[name], swapped[name] if name in names else original[name]), name
    write(OUT / 'swap_manifest.json', dict(source=str(GOAL), donor=str(BASE),
        swapped_names=names, non_adarms_tensors_verified_unchanged=len(original)-len(names),
        interpretation='Post-training SFT AdaRMS restoration, NOT training with AdaRMS frozen'))


def audit_checkpoint(path, kind):
    import torch
    from safetensors import safe_open
    state = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
    mismatches, compared, adapter_names = [], 0, []
    with safe_open(BASE, framework='pt', device='cpu') as f:
        keys = set(f.keys())
        for name, value in state.items():
            if '.lora_' in name:
                adapter_names.append(name)
                continue
            if name.startswith('value_head.'):
                continue
            if kind == 'adarms_only' and '.dense.' in name and 'norm' in name:
                continue
            normalized = name.replace('.base_layer.', '.')
            if normalized not in keys:
                mismatches.append(dict(name=name, reason='missing_in_base'))
                continue
            base = f.get_tensor(normalized).to(value.dtype)
            compared += 1
            if not torch.equal(value, base):
                mismatches.append(dict(name=name, reason='different_from_base'))
    write(OUT / f'audit_{kind}.json', dict(checkpoint=str(path), compared=compared,
          mismatches=mismatches, adapter_tensor_count=len(adapter_names), adapter_names=adapter_names))


def summarize(name, suite, result):
    text = (OUT / name / 'launcher.log').read_text()
    episodes = {}
    for t, i, ok in re.findall(r'\[libero eval\] task_id=(\d+), trial_id=(\d+), success=(True|False)', text):
        key = (int(t), int(i))
        assert key not in episodes, ('duplicate', key)
        episodes[key] = ok == 'True'
    assert len(episodes) == 500, len(episodes)
    tasks = {}
    for task in range(10):
        values = [v for (t, _), v in episodes.items() if t == task]
        assert len(values) == 50
        tasks[task] = dict(successes=sum(values), episodes=50, success_once=sum(values)/50)
    assert abs(sum(episodes.values())/500-result['metrics']['eval/success_once']) < 1e-5
    report = dict(suite=suite, tasks=tasks, metrics=result['metrics'])
    if suite == 'libero_spatial':
        for label, ids in [('seen7',[0,2,3,4,5,6,7]), ('heldout3',[1,8,9])]:
            report[label] = sum(tasks[t]['successes'] for t in ids)/(50*len(ids))
        report['caveat'] = '7/3 labels follow PPO/LoRA split; AdaRMS-only trained on all ten.'
    write(OUT / name / 'per_task.json', report)


def main():
    runtime.setup_environment()
    OUT.mkdir(exist_ok=True)
    diag.OUT = OUT
    lora80 = LORA_ROOT / 'global_step_80/actor/model_state_dict/full_weights.pt'
    lora50 = LORA_ROOT / 'global_step_50/actor/model_state_dict/full_weights.pt'
    for path in [BASE, GOAL, ADA, PPO, lora80, lora50]:
        assert path.is_file(), path
    suites = ['libero_spatial', 'libero_object', 'libero_goal']
    jobs = [('goal130_'+s, s, str(GOAL), 5, False) for s in suites[:2]]
    jobs += [('goal130_sftadarms_'+s, s, str(SWAP), 5, False) for s in suites]
    jobs += [('lora80_'+s, s, str(lora80), 3, True) for s in suites]
    jobs += [('adarms40_'+s, s, str(ADA), 3, False) for s in suites]
    jobs += [('ppo50_spatial', suites[0], str(PPO), 3, False),
             ('lora50_spatial', suites[0], str(lora50), 3, True),
             ('base_spatial', suites[0], '', 3, False),
             ('base_zero_lora_spatial', suites[0], '', 3, True)]
    jobs += [('ppo50_libero_object', 'libero_object', str(PPO), 5, False),
             ('ppo50_libero_goal', 'libero_goal', str(PPO), 5, False)]
    jobs = [(name,suite,checkpoint,3 if suite == 'libero_spatial' and not name.startswith('goal130') else 5 if suite != 'libero_spatial' else ns,lora)
            for name,suite,checkpoint,ns,lora in jobs]
    write(OUT / 'plan.json', dict(jobs=jobs, gpu=3, episodes_per_job=500,
          seed=42, generation=10, execution=5, parallel_envs=20,
          note='Single-GPU matched topology. New standalone evals, not pooled with old in-training scores.'))
    for index, (name,suite,checkpoint,ns,lora) in enumerate(jobs):
        if checkpoint == str(SWAP) and not (OUT/'swap_manifest.json').exists():
            status('preparing_ada_swap'); prepare_swap()
        if lora and checkpoint and not (OUT/'audit_lora.json').exists():
            status('auditing_lora'); audit_checkpoint(lora80, 'lora')
        if checkpoint == str(ADA) and not (OUT/'audit_adarms_only.json').exists():
            status('auditing_adarms'); audit_checkpoint(ADA, 'adarms_only')
        if not (OUT/name/'result.json').exists():
            while True:
                rows = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used',
                                                '--format=csv,noheader,nounits'],text=True)
                busy = any(int(r.split(',')[0]) == 3 and int(r.split(',')[1]) > 1000 for r in rows.splitlines())
                if not busy: break
                status('waiting_for_gpu3', job=name); time.sleep(30)
        overrides = ['actor.seed=42', '+rollout.model.openpi.action_horizon=10']
        if checkpoint:
            overrides += ['runner.ckpt_strict=true','rollout.model.add_value_head=true',
                          'rollout.model.openpi.add_value_head=true','rollout.model.openpi.value_after_vlm=true']
        if lora:
            overrides += ['rollout.model.is_lora=true','rollout.model.lora_rank=32',
                          '+rollout.model.lora_target=action_expert']
        status('evaluating', job=name, job_index=index+1, total=len(jobs))
        result = diag.evaluate((name,suite,5,ns,checkpoint),'3',trials=50,envs=20,
                               epochs=25,expected=500,timeout=21600,extra_overrides=overrides)
        summarize(name,suite,result)
        summaries = {p.parent.name:json.loads(p.read_text()) for p in OUT.glob('*/per_task.json')}
        write(OUT/'results.json',summaries)
    status('complete', total=len(jobs))


if __name__ == '__main__':
    try: main()
    except Exception as exc:
        status('failed', error=f'{type(exc).__name__}: {exc}')
        raise
