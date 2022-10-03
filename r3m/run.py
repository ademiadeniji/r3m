import subprocess

lr = 1e-4
# _err = f"logs/rlbenchrgb{lr}frontvar0scratchwckpt.log"
# _out = f"logs/rlbenchrgb{lr}frontvar0scratchwckpt.log"
_out = _err = f"logs/ood.log"
with open(_out,"wb") as out, open(_err,"wb") as err:
   command = f'python -u train_representation.py \
      hydra/launcher=local hydra/output=local agent.langweight=1.0 \
      agent.size=50 experiment=ood dataset=rlbench \
      doaug=rctraj agent.l1weight=0.00001 batch_size=16 \
      manifest_path=/shared/ademi_adeniji/r3m/rlbenchmanifestrgbfrontvar0.csv \
      wandbuser=adecyber wandbproject=r3mrlbench save_snapshot=True \
      lr={lr} eval_freq=1000 wandb=False load_snap=/shared/ademi_adeniji/r3m/snapshot_resnet50.pt'
   p = subprocess.Popen(command, shell=True, stdout=out, stderr=err, bufsize=0)
p.wait()
print('returncode', p.returncode)
# load_snap=/shared/ademi_adeniji/r3m/snapshot_resnet50.pt
