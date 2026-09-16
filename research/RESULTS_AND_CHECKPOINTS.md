# Results and checkpoint registry

Inventory snapshot: 2026-09-07. Paths are relative to the RLinf repository.
POST-CLEANUP: 24 intermediate saves deleted (454.86 GiB); 13 historical saves
retained. Inactive run paths below now resolve through symlinks into
`deprecated/results/`. Four historical downloaded models similarly moved into
`deprecated/checkpoints/`; LIBERO and ManiSkill SFT initializations stay in place.
The size/step inventory below is the BEFORE snapshot. Current retained steps:
Long PPO/CSD 10,90; forty-task PPO/CSD 10,20,80; gated10; newLong10; ManiSkill50.
Authoritative exact move/deletion list: archive_execution_20260907.json.
`historical` means preserved evidence, not deleted or scientifically invalid.
Raw run directories remain unchanged and are excluded from Git.

## Experiment families and canonical evidence

| Experiment | Status / purpose | Evidence root |
|---|---|---|
| Spatial 7/3 PPO | ACTIVE; current research direction | logs/20260907-spatial73 |
| ManiSkill memory-safe PPO | Stopped; reference for future small-task work | logs/maniskill_pi05_memsafe |
| ManiSkill ID SFT vs step50 | COMPLETE, two cells | logs/20260907-maniskill-cut-eval/results.json |
| ManiSkill OOD SFT vs step50 | COMPLETE, 24 cells | logs/20260907-maniskill-ood/results.json |
| Long 10-step PPO, gen/ex 10/10 | Historical short-run sanity reference | logs/20260906-pirl-long10-gpu03/step10_eval/results.json |
| Earlier Long10 attempt | Canceled before requested completion | logs/20260906-pirl-long10 |
| PPO130 attempts + official Plus-Spatial baseline eval | Historical; distinguish interrupted training from finished baseline eval | logs/20260906-pirl130 |
| Local step80 PPO/CSD/few-shot Plus-Spatial | Canceled/partial; not completed comparison | logs/20260906-local-plus-spatial |
| CSD diagnosis and confirmation | Historical correctness investigations | logs/20260905-csd-diagnosis/results.json; logs/20260905-csd-confirmation/results.json |
| Four-suite step80 evaluation | Historical PPO/CSD comparison; distinguish denoise3 from superseded denoise5 | logs/20260904-step80-eval-ns3-then-csdgate; logs/20260904-step80-eval-NUMSTEPS5-superseded |
| Old Long PPO/CSD step90 evaluations | Historical, different control settings from new Long10 | logs/20260901-libero-long-step90-safe-gpu4567-v2; logs/20260901-libero-long-step90-pro-gpu67 |
| Initial ManiSkill launch | Failed memory-capacity attempt, retain diagnosis | logs/maniskill_pi05_next |

Do not infer completion from a directory name. Preserve status, resolved config,
command and metric files. Earlier dated handoffs contain time-specific status.

## Completed ManiSkill scores

Each cell is success_once / success_at_end, percent; 320 episodes per model and
setting. Equal-weight average below covers the 12 OOD settings only.

| Setting | SFT | PPO step50 |
|---|---:|---:|
| ID | 58.1 / 55.9 | 69.1 / 51.6 |
| Instruct test | 55.9 / 53.4 | 73.4 / 54.4 |
| VisionImage test | 59.4 / 56.6 | 65.9 / 49.7 |
| VisionTexture03 test | 43.4 / 36.6 | 70.6 / 53.8 |
| VisionTexture05 test | 37.8 / 32.2 | 62.8 / 43.8 |
| VisionWhole03 test | 48.4 / 43.1 | 67.5 / 50.0 |
| VisionWhole05 test | 35.0 / 29.1 | 58.7 / 40.3 |
| MultiCarrot train | 30.0 / 28.4 | 41.9 / 31.6 |
| MultiCarrot test | 21.3 / 18.8 | 30.9 / 23.4 |
| MultiPlate train | 13.1 / 10.9 | 17.8 / 12.2 |
| MultiPlate test | 14.4 / 11.6 | 19.4 / 11.2 |
| PositionChangeTo test | 17.2 / 16.6 | 28.4 / 24.4 |
| Position test | 33.1 / 31.2 | 44.4 / 31.6 |
| OOD mean | 34.1 / 30.7 | 48.5 / 35.5 |

Protocol caveat: sampled horizon80, gen8/ex5, denoise4, mostly single receptacle;
not exhaustive paper coverage. `train` above denotes the variant's object split,
not PPO training on that distractor variant. Step60 peak and final live weights
were not saved; step50 is not claimed to be the globally best checkpoint.

## Saved training checkpoint roots

Append `global_step_<step>` to each root below. All 37 directories were observed
on disk; total allocated size 701.25 GiB. Each is approximately 18.95 GiB.
Existence/size is verified, not a full load or checksum-integrity audit.

| ID | Steps present | Proposed retained step |
|---|---|---|
| long-ppo | 10,20,30,40,50,60,70,80,90 | 90 |
| long-csd | 10,20,30,40,50,60,70,80,90 | 90 |
| forty-csd | 10,20,30,40,50,60,70,80 | 80 |
| forty-ppo-diag | 10,20,30,40,50,60,70,80 | 80 |
| forty-gated | 10 | 10 |
| long10-new | 10 | 10 |
| maniskill | 50 | 50 |

Exact roots:

```text
long-ppo: logs/20260828-libero_long_ppo_gpu67_gen10_exec5/libero_long_ppo_seed42_gpu67_gen10_exec5_20260828/checkpoints/
long-csd: logs/20260828-libero_long_ppo_csd_b1_gpu45_gen10_exec5/libero_long_ppo_csd_b1_seed42_gpu45_gen10_exec5_20260828/checkpoints/
forty-csd: logs/20260901-libero_40_ppo_csd_b1_gpu45_gen10_exec5/libero_40_ppo_csd_b1_seed42_gpu45_gen10_exec5_20260901/checkpoints/
forty-ppo-diag: logs/20260902-libero_40_ppo_csddiag_gpu67_gen10_exec5/libero_40_ppo_csddiag_seed42_gpu67_gen10_exec5_20260902/checkpoints/
forty-gated: logs/20260905-libero_40_ppo_csdgate_padv_gpu45_gen10_exec5/libero_40_ppo_csdgate_padv_seed42_gpu45_gen10_exec5_20260905/checkpoints/
long10-new: logs/20260906-pirl-long10-gpu03/train/pi05_ppo_long_paper_seed42_10steps_20260906_gpu03/checkpoints/
maniskill: logs/maniskill_pi05_memsafe/pi05_maniskill_flownoise_memsafe/checkpoints/
```

The similarly named 20260901 40-task PPO directory is not the saved checkpoint
root used for the later PPO-diagnostic comparison. Do not merge these identities.
Spatial has not reached its first scheduled save at this inventory snapshot;
future checkpoints are protected regardless of this static list.

## Downloaded initialization/reference models: keep for now

| Directory under checkpoints/ | Approximate allocated size | Purpose |
|---|---:|---|
| RLinf-Pi05-LIBERO-SFT | 7 GiB | Active Spatial initialization / SFT reference |
| RLinf-Pi05-ManiSkill-25Main-SFT | 7 GiB | ManiSkill initialization / SFT reference |
| RLinf-Pi05-LIBERO-130-fullshot-SFT | 7 GiB | Official full-shot comparison |
| RLinf-Pi05-PPO-LIBERO-130 | 8 GiB | Official RL comparison |
| RLinf-Pi0-PPO-LIBERO-spatial | 7.6 GiB | Separate pi0 reference, not pi0.5 |
| lerobot-pi05-libero-base | 14 GiB | Alternative SFT implementation/reference |

Do not assume similarly named files are duplicate weights. Before any later
deletion verify the exact upstream revision, local modifications, accessibility,
normalization/config files and whether a live/queued job references the path.
