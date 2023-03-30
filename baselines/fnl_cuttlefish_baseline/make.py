# // Copyright (c) Microsoft Corporation.
# // Licensed under the MIT license.
import os
import pdb
import sys
from copy import deepcopy
from operator import itemgetter


LOTTERY = "EigenDamage-Pytorch"
RESNET = 'pytorch_resnet_cifar10'
TENSOR = "deficient-efficient"
TRANSFORMER = "Transformer-PyTorch"
SCRIPTS = "generated-scripts"


def parameter_count(model):

    return sum(p.numel() for p in model.parameters())


def compress_model(model, compress, target, lower=0.0, upper=1.0, tol=1E-6, **kwargs):

    origpar = parameter_count(model)
    middle = 0.5 * (upper + lower)
    compressed = compress(deepcopy(model), middle, **kwargs)
    ratio = parameter_count(compressed) / origpar
    if upper - lower < tol or abs(ratio - target) < tol:
        return compressed, middle

    if ratio < target:
        refine, scale = compress_model(model, compress, target, lower=middle, upper=upper, tol=tol, **kwargs)
    else:
        refine, scale = compress_model(model, compress, target, lower=lower, upper=middle, tol=tol, **kwargs)
    if abs(parameter_count(refine) / origpar - target) < abs(ratio - target):
        return refine, scale
    return compressed, middle


def generate_scripts(settings, folder, reverse=False, trials=1, devices=1, results='results', one_gpu=False, one_trial=False):

    totals = {device: 0.0 for device in range(devices)}
    info = {device: [] for device in range(devices)}
    names = {}
    for command, specs, total, name in sorted(settings, key=itemgetter(2), reverse=reverse):
        for trial in range(1, trials+1 if one_trial else 2):
            device = min(totals.items(), key=itemgetter(1))[0]
            totals[device] += (1 if one_trial else trials) * total
            info[device].append((command, 
                                 specs, 
                                 str(trial if one_trial else 1), 
                                 str(trial if one_trial else trials)))
            names[device] = name + ('--'+str(trial) if one_trial else '')

    scriptdir = os.path.join(folder, SCRIPTS)
    os.makedirs(scriptdir, exist_ok=True)
    print('\nGenerating Scripts in', scriptdir)
    for device, settings in info.items():
        if not settings:
            continue
        scriptname = names[device] + '.sh'
        #scriptname = 'device' + str(device) + '.sh'
        with open(os.path.join(folder, SCRIPTS, scriptname), 'w') as f:
            f.write('#!/bin/bash\n\n\nGPU=' + str(0 if one_gpu else device))
            f.write('\nexport PYTHONPATH="..:"$PYTHONPATH\nRESULTS='+results+'\n\n')
            for command, specs, a, b in reversed(settings) if reverse else settings:
                f.write('\nfor TRIAL in `seq '+a+' '+b+'` ; do\n')
                for spec in specs:
                    f.write('\t'+spec+'\n')
                f.write('\t'+command+'\n')
                f.write('done\n')
        print('\tEstimated Device', device, 'Time:', totals[device] / 60.0, 'hours')



LOTTERY_COMMAND = ['python main_pretrain.py',
                   '--auto-resume',
                   '--dataset $DATA',
                   '--network $MODEL',
                   '--weight_decay $WD',
                   '--depth $DEPTH',
                   '--device $GPU',
                   '--epoch $EPOCH',
                   '--target-ratio $RATIO',
                   '--log_dir $LOGDIR']

LOTTERY_MINUTES = lambda data, network: {'resnet_32_x2': 80, 'vgg_19_nfc': 36}[network] if 'cifar' in data else {'resnet_32_x2': 630, 'vgg_19_nfc': 200}[network]

def lottery_command(network='resnet_32_x2',
                    data='cifar10',
                    ratio='0.1',
                    spectral=False,
                    frob=False):

    command = ' '.join(LOTTERY_COMMAND)
    specs = ['DATA=' + data,
             'RATIO='+ratio]
    if network == 'resnet_32_x2':
        specs.extend(['MODEL=resnet',
                      'DEPTH=32',
                      'WD=1E-4'])
    elif network == 'vgg_19_nfc':
        specs.extend(['MODEL=vgg',
                      'DEPTH=19',
                      'WD=2E-4'])
    else:
        raise(NotImplementedError)

    if data == 'cifar10' or data == 'cifar100':
        specs.append('EPOCH=200')
    elif data == 'tiny_imagenet':
        specs.append('EPOCH=200')
    else:
        raise(NotImplementedError)

    methods = []
    if spectral:
        command += ' --spectral'
        methods.append('spectral')
    if frob:
        command += ' --wd2fd'
        methods.append('frob')
    name = '+'.join(methods) if methods else 'reglr'

    specs.append('LOGDIR='+os.path.join('$RESULTS',
                                        '-'.join([network, '$DATA']),
                                        '$RATIO',
                                        name,
                                        '$TRIAL'))

    return command, specs, LOTTERY_MINUTES(data, network), '-'.join([data, network, ratio, name])

def lottery_settings():

    settings = []
    for network in ['resnet_32_x2', 'vgg_19_nfc']:
        for data in ['cifar10', 'cifar100', 'tiny_imagenet']:
            for ratio in [0.0, 0.05, 0.1, 0.15] if 'resnet' in network and 'tiny' in data else [0.0, 0.02, 0.05, 0.1]:
                for spectral in [False, True] if ratio else [False]:
                    for frob in [False, True] if ratio else [False]:
                        settings.append(lottery_command(network=network, data=data, ratio=str(ratio), 
                                                        spectral=spectral, frob=frob))
    return settings


TENSOR_COMMAND = ['python main.py',
                  '$DATA',
                  'teacher',
                  '--wrn_depth 28',
                  '--wrn_width 10',
                  '--GPU $GPU',
                  '--epochs 200',
                  '--conv $CONV',
                  '-t $LOGDIR']
TENSOR_MINUTES = lambda ratio: 180 if float(ratio) < 1.0 else 270

def tensor_command(decomp='Conv',
                   ratio='0.1',
                   spectral=False,
                   decay='reglr'):

    command = ' '.join(TENSOR_COMMAND)
    specs = ['DATA=cifar10',
             'RATIO='+ratio,
             'CONV='+decomp]
    if decomp == 'Conv':
        command += ' --target-ratio $RATIO'
    else:
        specs[-1] += '_$RATIO'

    methods = []
    if spectral:
        command += ' --spectral'
        methods.append('spectral')
    if decay == 'crs':
        methods.append(decay)
    elif decay == 'frob':
        command += ' --wd2fd'
        methods.append(decay)
    elif decay == 'reglr':
        command += ' --nocrswd'
    else:
        raise(NotImplementedError)
    name = '+'.join(methods) if methods else decay

    specs.append('LOGDIR='+os.path.join('$RESULTS',
                                        '-'.join([decomp, 'wrn_28_10', '$DATA']),
                                        '$RATIO',
                                        name,
                                        '$TRIAL'))

    return command, specs, TENSOR_MINUTES(ratio), '-'.join([decomp, ratio, name])
    
def tensor_settings():

    settings = []
    for decomp in ['Conv', 'TensorTrain', 'Tucker']:
        for ratio in {'Conv': [0.0, 0.01667, 0.06667], 'TensorTrain': [0.234, 0.705], 'Tucker': [0.211, 0.68]}[decomp]:
            for spectral in [False, True] if ratio else [False]:
                for decay in ['reglr', 'crs', 'frob'] if ratio else ['reglr']:
                    settings.append(tensor_command(decomp=decomp, ratio=str(ratio), 
                                                   spectral=spectral, decay=decay))
    return settings


TRANSFORMER_COMMAND = [['python train.py $DATA',
                        '--arch $MODEL',
                        '--clip-norm 0.1',
                        '--dropout 0.2',
                        '--max-tokens 4000',
                        '--criterion label_smoothed_cross_entropy',
                        '--label-smoothing 0.1',
                        '--lr-scheduler inverse_sqrt',
                        '--lr 0.25',
                        '--optimizer nag',
                        '--warmup-init-lr 0.25',
                        '--warmup-updates 4000',
                        '--max-update 100000',
                        '--no-epoch-checkpoints',
                        '--distributed-world-size 1',
                        '--device-id $GPU',
                        '--seed $TRIAL',
                        '--save-dir $LOGDIR',
                        '--rank-scale $SCALE'],
                       ['python generate.py $DATA',
                        '--batch-size 128',
                        '--beam 5',
                        '--remove-bpe',
                        '--quiet',
                        '--path $LOGDIR/checkpoint_best.pt',
                        '--dump $LOGDIR/bleu.log',
                        '--rank-scale $SCALE']]
TRANSFORMER_MINUTES = 240

def transformer_command(scale='0.0',
                        decay='0.0',
                        spectral=False,
                        spectral_qk=False,
                        spectral_ov=False,
                        frob=False,
                        frob_qk=False,
                        frob_ov=False):

    commands = [' '.join(command) for command in TRANSFORMER_COMMAND]
    specs = ['DATA=data-bin/iwslt14.tokenized.de-en',
             'MODEL=transformer_small',
             'DECAY='+decay,
             'SCALE='+scale]

    methods = []
    for name, command, group in [('spectral', 'spectral', [spectral, spectral_qk, spectral_ov]),
                                 ('frob', 'wd2fd', [frob, frob_qk, frob_ov])]:
        if any(group):
            methods.append(name)
        for modify, suffix, option in zip(['', '-quekey', '-outval'],
                                           ['_lin', '_qk', '_ov'],
                                           group):
            if option:
                commands[0] += ' --' + command + modify
                methods[-1] += suffix
    if any(group):
        commands[0] += ' --frobenius-decay $DECAY'
    else:
        commands[0] += ' --weight-decay $DECAY'
    name = '+'.join(methods) if methods else 'reglr'

    specs.append('LOGDIR='+os.path.join('$RESULTS',
                                        '-'.join(['$MODEL', 'iwslt14_deen']),
                                        '$SCALE',
                                        name,
                                        '$DECAY',
                                        '$TRIAL'))

    return '\n\t'.join(commands), specs, TRANSFORMER_MINUTES, '-'.join([name, 'scale_'+scale, 'decay_'+decay])

def transformer_settings():

    settings = []
    for scale in [0.0, 0.25, 0.5]:
        for decay in [0.0, 1E-6, 5E-6, 1E-5, 5E-5, 1E-4] if scale else [0.0, 1E-6, 5E-6, 1E-5, 5E-5, 1E-4, 5E-4, 1E-3]:
            for frob_ov in [False, True] if decay else [False]:
                for frob in [False, True] if decay and scale else [False]:
                    for frob_qk in [False, True] if frob_ov else [False]:
                        for spectral in [False, True] if scale else [False]:
                            for spectral_ov in [False, True] if 0.0 <= decay <= 1E-4 else [False]:
                                for spectral_qk in [spectral_ov]: 
                                    settings.append(transformer_command(scale=str(scale), decay=str(decay),
                                                                        spectral=spectral, spectral_qk=spectral_qk, spectral_ov=spectral_ov,
                                                                        frob=frob, frob_qk=frob_qk, frob_ov=frob_ov))
    return settings


RESNET_COMMAND = ['python trainer.py',
                  '--data $DATA',
                  '--arch $MODEL',
                  '--device $GPU',
                  '--weight-decay $DECAY',
                  '--rank-scale $SCALE',
                  '--seed $TRIAL',
                  '--save-dir $LOGDIR']
RESNET_MINUTES = lambda depth, norm: (1+2*int(norm)) * {20: 50, 32: 60, 44: 80, 56: 90, 110: 180}[int(depth)]

def resnet_command(data='cifar10',
                   scale='0.0',
                   decay='1E-4',
                   depth='32',
                   spectral=False,
                   frob=False,
                   square=False,
                   norm=False,
                   residual=False):

    specs = ['DEPTH='+depth,
             'MODEL=resnet$DEPTH',
             'DECAY='+decay,
             'SCALE='+scale,
             'DATA='+data]
    
    if norm:
        commands = [' '.join(RESNET_COMMAND) for _ in range(3)]
        commands[0] += '/none/$TRIAL --no-frob'
        commands[1] += '/frob/$TRIAL --dump-frobnorms'
        commands[2] += '/norm/$TRIAL --no-frob --normalize $LOGDIR/frob/$TRIAL/frobnorms.tensor'
        name = 'norm+spectral' if spectral else 'norm+reglr'
        for i in range(3):
            commands[i] += ' --wd2fd'
            if spectral:
                commands[i] += ' --spectral'
        command = '\n\t'.join(commands)

    else:
        command = ' '.join(RESNET_COMMAND)
        methods = []
        if spectral:
            command += ' --spectral'
            methods.append('spectral')
        if frob:
            command += ' --wd2fd'
            methods.append('frob')
        if square:
            command += ' --square'
            methods.append('square')
            if residual:
                specs.append('STD=0.01')
                command += ' --residual $STD'
                methods[-1] += '$STD'
        name = os.path.join('$DECAY', '+'.join(methods) if methods else 'reglr', '$TRIAL')

    specs.append('LOGDIR='+os.path.join('$RESULTS',
                                        '-'.join(['$MODEL', '$DATA']),
                                        '$SCALE',
                                        name))
    return command, specs, RESNET_MINUTES(depth, norm), \
            '-'.join([data, 'resnet'+depth, 'scale_'+scale, name.replace('$DECAY', 'decay_'+decay).replace('$TRIAL', '').replace('$STD', '').replace('/', '-')])[:-1]

def resnet_settings():

    settings = []
    for data in ['cifar10', 'cifar100']:
        for depth in [20, 32, 44, 56, 110]:
            settings.append(resnet_command(data=data, depth=str(depth)))
            for logscale in range(-3, 3):
                scale = str(round(3.0 ** logscale, 3))
                if depth in {32, 56, 110}: 
                    for square in [False] if logscale else [False, True]:
                        for spectral in [False, True] if logscale <= 0 else [False]:
                            for frob in [False, True]:
                                settings.append(resnet_command(data=data, scale=scale, depth=str(depth), 
                                                               spectral=spectral, frob=frob, 
                                                               square=square, residual=False))
                elif logscale == -2 and depth == 20 and data == 'cifar10':
                    for decay in [5E-6, 1E-5, 5E-5, 5E-4, 1E-3, 5E-3]:
                        for frob in [False, True]:
                            settings.append(resnet_command(data=data, scale=scale, depth=str(depth), 
                                                           decay=str(decay), frob=frob))
                    for spectral in [False, True]:
                        settings.append(resnet_command(data=data, scale=scale, depth=str(depth), 
                                                       spectral=spectral, norm=True))
    return settings


if __name__ == '__main__':

    kwargs = {'trials': 3, 
              'devices': 10000,
              'results': 'results',
              'one_gpu': True,
              'one_trial': False}
    generate_scripts(lottery_settings(), LOTTERY, reverse=True, **kwargs)
    generate_scripts(tensor_settings(), TENSOR, **kwargs)
    generate_scripts(transformer_settings(), TRANSFORMER, **kwargs)
    generate_scripts(resnet_settings(), RESNET, reverse=True, **kwargs)
