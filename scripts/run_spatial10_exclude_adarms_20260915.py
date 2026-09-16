"""Launch all-10 Spatial PPO with AdaRMS frozen on GPUs0-3."""
import argparse
import json
import os
import shutil
import subprocess

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import run_pirl130_20260906 as runtime

OUT = runtime.ROOT / 'logs/20260915-spatial10-ppo-exclude-adarms'
CONFIG = 'libero_spatial10_ppo_exclude_adarms_20260915'
OVERRIDES = ['runner.max_steps=100', 'runner.save_interval=10',
             'runner.val_check_interval=10']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    runtime.setup_environment()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    with initialize_config_dir(config_dir=str(runtime.ROOT/'examples/embodiment/config'),
                               version_base='1.1'):
        cfg=compose(config_name=CONFIG, overrides=OVERRIDES)
    assert cfg.cluster.component_placement['actor,env,rollout']=='0-3'
    assert cfg.actor.model.trainability_mask=='exclude_adarms'
    assert cfg.actor.model.openpi.train_expert_only and not cfg.actor.model.is_lora
    assert not cfg.actor.fsdp_config.use_orig_params
    assert cfg.actor.model.num_steps==3
    assert cfg.actor.model.num_action_chunks==5 and cfg.actor.model.openpi.action_horizon==10
    assert cfg.actor.optim.lr==5e-6 and cfg.actor.optim.value_lr==1e-4
    assert cfg.actor.optim.clip_grad==1.0 and cfg.actor.optim.adam_eps==1e-8
    assert cfg.algorithm.update_epoch==1 and not cfg.algorithm.csd_enabled
    assert cfg.actor.global_batch_size==2048 and cfg.actor.micro_batch_size==16
    assert cfg.env.train.total_num_envs==64 and cfg.env.train.rollout_epoch==8
    assert cfg.env.train.max_episode_steps==240
    assert list(cfg.env.train.task_id_filter)==list(range(10))
    assert list(cfg.env.eval.task_id_filter)==list(range(10))
    assert cfg.env.eval.num_trials_per_task==50
    assert cfg.runner.save_interval==cfg.runner.val_check_interval==10
    assert cfg.runner.max_steps==100 and cfg.runner.resume_dir is None
    assert cfg.runner.ckpt_path is None
    assert 'wandb' in cfg.runner.logger.logger_backends
    assert (runtime.ROOT/'checkpoints/RLinf-Pi05-LIBERO-SFT/model.safetensors').is_file()
    print('Validated: fresh SFT, all10 Spatial, PPO100, AdaRMS frozen, save/eval10, GPUs0-3',flush=True)
    if not args.launch:return
    for line in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used',
                                        '--format=csv,noheader,nounits'],text=True).splitlines():
        gpu,memory=map(int,line.split(','))
        if gpu<4 and memory>1000:raise RuntimeError(f'GPU{gpu} occupied')
    assert shutil.disk_usage(runtime.ROOT).free>250*2**30, 'Insufficient checkpoint headroom'
    OUT.mkdir(exist_ok=False)
    OmegaConf.save(cfg, OUT/'resolved.yaml', resolve=True)
    (OUT/'manifest.json').write_text(json.dumps(dict(config=CONFIG, initialization='fresh original SFT',
        trainability='action expert except AdaRMS; value head trained; VLM frozen',
        gpus=[0,1,2,3], max_steps=100, save_interval=10, eval_interval=10,
        eval_episodes=500),indent=2))
    runtime.OUT=OUT
    runtime.sanity_evaluation=lambda:None
    runtime.train(config_name=CONFIG,gpus=(0,1,2,3),extra_overrides=OVERRIDES,min_free_gib=40)


if __name__=='__main__':
    main()
