import subprocess

lr = 1e-5
_out = f"logs/rlbenchftlr{lr}saveevery5k.log"
_err = f"logs/rlbenchftlr{lr}saveevery5k_err.log"
with open(_out,"wb") as out, open(_err,"wb") as err:
   command = f'python -u train_representation.py \
      hydra/launcher=local hydra/output=local agent.langweight=1.0 \
      agent.size=50 experiment=rlbenchftlr{lr}saveevery5k dataset=rlbench \
      doaug=rctraj agent.l1weight=0.00001 batch_size=16 \
      datapath=/shared/mandi/all_rlbench_data \
      wandbuser=adecyber wandbproject=r3mrlbench save_snapshot=True \
      load_snap=/shared/ademi_adeniji/r3m/snapshot_resnet50.pt lr={lr} \
      eval_freq=5000'
   p = subprocess.Popen(command, shell=True, stdout=out, stderr=err, bufsize=0)

p.wait()
print('returncode', p.returncode)

# _out = f"logs/r3mtestcont.log"
# _err = f"logs/r3mtestcont_err.log"
# with open(_out,"wb") as out, open(_err,"wb") as err:
#    command = f'python -u train_representation.py \
#       hydra/launcher=local hydra/output=local agent.langweight=1.0 \
#       agent.size=50 experiment=rltestcont dataset=rlbench \
#       doaug=rctraj agent.l1weight=0.00001 batch_size=16 \
#       datapath=/shared/mandi/all_rlbench_data \
#       wandbuser=adecyber wandbproject=rlbenchtest save_snapshot=True \
#       load_snap=/shared/ademi_adeniji/r3m/r3m/r3moutput/train_representation/2022-04-23_17-28-19/snapshot_60000.pt'
#    p = subprocess.Popen(command, shell=True, stdout=out, stderr=err, bufsize=0)