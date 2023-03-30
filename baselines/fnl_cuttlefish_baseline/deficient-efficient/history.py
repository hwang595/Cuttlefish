# opens checkpoints and prints the commands used to run each
import torch
import os
import argparse

parser = argparse.ArgumentParser(description='Inspect saved checkpoints')
parser.add_argument('--match', type=str, default=None, help='Filter checkpoints by keyword.')

if __name__ == '__main__':
    args = parser.parse_args()
    ckpt_paths = os.listdir("checkpoints")
    # filter for search term
    if args.match is not None:
        ckpt_paths = [c for c in ckpt_paths if args.match in c]
    for p in ckpt_paths:
        try:
            ckpt = torch.load("checkpoints/"+p)
            if 'args' in ckpt.keys():
                print(p + " (%i-%.2f) "%(ckpt['epoch'], ckpt['val_errors'][-1]) + ":  " + " ".join(ckpt['args']))
        except:
            pass
