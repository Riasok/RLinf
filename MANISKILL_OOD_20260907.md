# ManiSkill OOD comparison queue

After the ongoing ID SFT/step50 comparison finishes and releases GPUs4–7,
run12 settings from pi_RL Table7 for each model (24cells,320episodes each).
Vision/language: Instruct,VisionImage,VisionTexture03/05,VisionWhole03/05,all test.
Semantic: MultiCarrot and MultiPlate,each train and test. Execution:
PositionChangeTo and Position,test. Environment IDs use PutOnPlateInScene25
prefix and -v1 suffix. The train suffix on a confounder variant is still an OOD
test of the added distractor mechanism, not the unmodified training environment.

Maintain gen8/ex5,denoise4,ODE,horizon80,seed0,320envs across fourGPUs,oneepoch.
Separate success_once and success_at_end per cell. Compare models within cells.
This is sampled current-code evaluation,not exact paper reproduction:
main environment uses use_multiple_plates=false (one receptacle),whereas paper
describes16objects×17receptacles×16scenes=4352combinations. Some variants,
notably MultiPlate,override that restriction internally. No training restarts.
Step50 is the only saved trained checkpoint; peak60 and final live weights absent.

Queue logs/20260907-maniskill-ood/plan.json and status.json; cells have their own
config,commands,logs,W&B and results. The wrapper stops on failed cells instead
of silently omitting them. Registry validation occurs before waiting; actual
variant GPU instantiation is checked when each cell launches,not pre-certified.
Source https://arxiv.org/html/2510.25889 Table7 and AppendixC.2/D.1.
