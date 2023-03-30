# open schedule json, then search for which machines the longest progressed job
# has run on
import json
import sys
import os
import torch
import subprocess
from subprocess import PIPE
from collections import OrderedDict

from funcs import what_conv_block
from models.wide_resnet import WideResNet, WRN_50_2
from models.darts import DARTS
from count import measure_model

from tqdm import tqdm

with open('machine_list.json', 'r') as f:
    # list of strings with "hostname:path" where the deficient-efficient repos
    # can be found
    machines = json.load(f)

def ckpt_name(experiment):
    if '-s' in experiment:
        prefix = '-s'
    else:
        prefix = '-t'
    ckpt_idx = [i for i, arg in enumerate(experiment) if arg == prefix][0]+1
    return experiment[ckpt_idx]

def parse_name(path):
    monthday = path.split(".")[-2]
    path = path.split('.')[1:] # split off part containing settings
    # index to cut out for settings
    idx = [i for i,s in enumerate(path) if monthday == s or 'student' == s][0]
    method, setting = (".".join(path[:idx])).split("_") # this is just the settings string now
    return 'student' in path, method, setting

def parse_checkpoint(ckpt_name, ckpt_contents):
    results = OrderedDict()
    results['epoch'] = ckpt_contents['epoch']
    results['val_errors'] = [float(x) for x in ckpt_contents['val_errors']]
    results['train_errors'] = [float(x) for x in ckpt_contents['train_errors']]
    # hard part: count parameters by making an instance of the network
    network = {'wrn_28_10': 'WideResNet', 'darts': 'DARTS', 'wrn_50_2': 'WRN_50_2'}[ckpt_name.split(".")[0]]
    h,w = {'WideResNet': (32,32), 'DARTS': (32,32), 'WRN_50_2': (224,224)}[network]
    SavedConv, SavedBlock = what_conv_block(ckpt_contents['conv'],
            ckpt_contents['blocktype'], ckpt_contents['module'])
    model = build_network(SavedConv, SavedBlock, network)
    flops, params = measure_model(model, h, w)
    assert params == sum([p.numel() for p in model.parameters()])
    results['no_params'] = params
    results['flops'] = flops
    results['settings'] = parse_name(ckpt_name)
    results['scatter'] = (params, results['val_errors'][-1], results['train_errors'][-1], results['epoch'], flops)
    return results

# instance the model
def build_network(Conv, Block, network):
    if network == 'WideResNet':
        return WideResNet(28, 10, Conv, Block,
                num_classes=10, dropRate=0)
    elif network == 'WRN_50_2':
        return WRN_50_2(Conv)
    elif network == 'DARTS':
        return DARTS(Conv, num_classes=10, drop_path_prob=0., auxiliary=False)

def keep_oldest(collated, ckpt_name, ckpt_contents):
    # if the checkpoint already exists in collated,
    # keep it if it's run for more epochs
    ckpt = parse_checkpoint(ckpt_name, ckpt_contents)
    try:
        existing_epochs = collated[ckpt_name]['epoch']
    except KeyError:
        # doesn't exist yet so return
        return ckpt
    if int(existing_epochs) < int(ckpt['epoch']):
        return ckpt
    else:
        return collated[ckpt_name]

def main():
    try:
        # read the schedule from json
        json_path = sys.argv[1]
        with open(json_path, "r") as f:
            schedule = json.load(f)

        # prepare directory
        if not os.path.exists("collate"):
            os.mkdir("collate")
        else:
            # clean up directory
            old_ckpts = os.listdir("collate")
            for c in old_ckpts:
                os.remove(os.path.join("collate", c))

        # make a list of all the checkpoint files we need to check
        checkpoints = []
        for e in schedule:
            checkpoints.append(ckpt_name(e)+".t7")
        # look for these checkpoints on every machine we know about
        collated = []
        for m in tqdm(machines, desc='machine'):
            # connect to the remote machine
            hostname, directory = m.split(":")
            checkpoint_dir = os.path.join(directory, "checkpoints")
            completed = subprocess.run(f"ssh {hostname} ls {checkpoint_dir}".split(" "), stdout=PIPE, stderr=PIPE)
            checkpoints_on_remote = completed.stdout.decode().split("\n")

            # look for overlap between that and the checkpoints we care about
            overlap = list(set(checkpoints_on_remote) & set(checkpoints))
            for checkpoint in tqdm(overlap, desc="copying"):
                checkpoint_loc = os.path.join(checkpoint_dir, checkpoint)
                checkpoint_dest = f"collate/{hostname}.{checkpoint}"
                if not os.path.exists(checkpoint_dest):
                    subprocess.run(f"scp {hostname}:{checkpoint_loc} {checkpoint_dest}".split(" "), stdout=PIPE, stderr=PIPE)

    except IndexError:
        pass

    # iterate over copied files
    collated = OrderedDict()
    copied = os.listdir("collate")
    for checkpoint in tqdm(copied, desc='Opening checkpoints'):
        checkpoint_loc = os.path.join("collate", checkpoint)
        hostname = checkpoint.split(".")[0]
        checkpoint_name = ".".join(checkpoint.split(".")[1:])
        checkpoint_contents = torch.load(checkpoint_loc)
        collated[checkpoint_name] = keep_oldest(collated, checkpoint_name, checkpoint_contents)

    for k in collated:
        print(k, collated[k]['epoch'], collated[k]['val_errors'][-1])

    with open("collated.json", "w") as f:
        f.write(json.dumps(collated, indent=2))

if __name__ == "__main__":
    main()

