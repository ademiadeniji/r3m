import subprocess

lr = 1e-4
_out = _err = f"logs/drawerfromego4dpretrain.log"
with open(_out,"wb") as out, open(_err,"wb") as err:
   command = f'python -u train_representation.py ' + \
      f'hydra/launcher=local hydra/output=local agent.langweight=1.0 ' + \
      f'agent.size=50 experiment=drawerfromego4dpretrain dataset=something2something ' + \
      f'doaug=rctraj agent.l1weight=0.00001 batch_size=16 ' +\
      f'manifest_path=/home/ademi_adeniji/r3m/drawer_manifest.csv ' + \
      f'wandbuser=adecyber wandbproject=r3mrlbench save_snapshot=True ' + \
      f'lr={lr} eval_freq=1000 wandb=True load_snap=/shared/ademi_adeniji/r3m/snapshot_resnet50.pt'
   p = subprocess.Popen(command, shell=True, stdout=out, stderr=err, bufsize=0)
   print(command)
p.wait()
print('returncode', p.returncode)
# load_snap=/shared/ademi_adeniji/r3m/snapshot_resnet50.pt
