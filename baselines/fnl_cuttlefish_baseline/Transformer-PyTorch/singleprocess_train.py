#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import collections
import itertools
import json
import os
import pdb
import math
import torch
from torch import nn
from torch.nn import functional as F

from fairseq import criterions, data, models, options, progress_bar
from fairseq.meters import AverageMeter, StopwatchMeter
from fairseq.trainer import Trainer
from tensorboardX import SummaryWriter

try:
    from frob import FactorizedLinear, batch_spectral_init, frobenius_norm, patch_module, non_orthogonality
except ImportError:
    print("Failed to import factorization")


class FactorizedEmbedding(FactorizedLinear):

    def __init__(self, embedding, **kwargs):
        
        embedding.bias = None
        super().__init__(embedding, **kwargs)
        self.kwargs = {'padding_idx': embedding.padding_idx,
                       'max_norm': embedding.max_norm,
                       'norm_type': embedding.norm_type,
                       'scale_grad_by_freq': embedding.scale_grad_by_freq,
                       'sparse': embedding.sparse}
        self.embedding = embedding
        self.max_positions = embedding.max_positions if hasattr(embedding, 'max_positions') else None
        self.embedding.weight = self.U

    def forward(self, *x):

        return F.linear(self.embedding(*x), self.VT.T)


def patch_transformer(args, model):

    for coder in [model.encoder, model.decoder]:
        for block in coder.ffn_blocks:
            for name in ['fc1', 'fc2']:
                patch_module(block, name, FactorizedLinear,
                             rank_scale=args.rank_scale,
                             init='spectral' if args.spectral else lambda X: nn.init.uniform_(X, -0.1, 0.1))
        for name in ['embed_tokens', 'embed_positions']:
            patch_module(coder, name, FactorizedEmbedding,
                         rank_scale=args.rank_scale,
                         init='spectral' if args.spectral else lambda X: nn.init.normal_(X, 0, 0.1))
    patch_module(coder, 'out_embed', FactorizedLinear,
                 rank_scale=args.rank_scale,
                 init='spectral' if args.spectral else lambda X: nn.init.uniform_(X, -0.1, 0.1))


def spectral_init(args, model):

    for module in model.modules():
        if args.spectral_quekey and hasattr(module, 'quekey'):
            batch_spectral_init(*module.quekey.get_UVT())
        if args.spectral_outval and hasattr(module, 'outval'):
            batch_spectral_init(*module.outval.get_UVT())


def main(args):
    print(args)

    if not torch.cuda.is_available():
        raise NotImplementedError('Training on CPU is not supported')
    torch.cuda.set_device(args.device_id)
    torch.manual_seed(args.seed)

    # Load dataset
    splits = ['train', 'valid']
    if data.has_binary_files(args.data, splits):
        dataset = data.load_dataset(
            args.data, splits, args.source_lang, args.target_lang)
    else:
        dataset = data.load_raw_text_dataset(
            args.data, splits, args.source_lang, args.target_lang)
    if args.source_lang is None or args.target_lang is None:
        # record inferred languages in args, so that it's saved in checkpoints
        args.source_lang, args.target_lang = dataset.src, dataset.dst
    print('| [{}] dictionary: {} types'.format(dataset.src, len(dataset.src_dict)))
    print('| [{}] dictionary: {} types'.format(dataset.dst, len(dataset.dst_dict)))
    for split in splits:
        print('| {} {} {} examples'.format(args.data, split, len(dataset.splits[split])))

    # Build model and criterion
    model = models.build_model(args, dataset.src_dict, dataset.dst_dict)
    if 0.0 < args.rank_scale < 1.0:
        patch_transformer(args, model)
        if args.wd2fd:
            no_decay, skiplist = ['fc1', 'fc2', 'embed_tokens', 'embed_positions', 'out_embed'], []
        else:
            no_decay, skiplist = [], ['fc1', 'fc2', 'embed_tokens', 'embed_positions', 'out_embed']
    else:
        no_decay, skiplist = [], []
    spectral_init(args, model)

    criterion = criterions.build_criterion(args, dataset.src_dict, dataset.dst_dict)
    print('| model {}, criterion {}'.format(args.arch, criterion.__class__.__name__))
    print('| num. model params: {}'.format(sum(p.data.numel() for p in model.parameters())))

    # Build trainer
    no_decay, skiplist = [], []
    if args.wd2fd_quekey:
        no_decay.extend(['_query.weight', '_key.weight'])
    else:
        skiplist.append('quekey')
    if args.wd2fd_outval:
        no_decay.extend(['_value.weight', 'output_perform.weight'])
    else:
        skiplist.append('outval')

    trainer = Trainer(args, model, criterion, skiplist=skiplist, no_decay=no_decay)
    print('| training on {} GPUs'.format(args.distributed_world_size))
    print('| max tokens per GPU = {} and max sentences per GPU = {}'.format(
        args.max_tokens,
        args.max_sentences,
    ))

    # Load the latest checkpoint if one is available
    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.save_dir, args.restore_file)
    extra_state = trainer.load_checkpoint(checkpoint_path)
    if extra_state is not None:
        epoch = extra_state['epoch']
        batch_offset = extra_state['batch_offset']
        print('| loaded checkpoint {} (epoch {})'.format(checkpoint_path, epoch))
        if batch_offset == 0:
            trainer.lr_step(epoch)
            epoch += 1
    else:
        epoch, batch_offset = 1, 0

    if args.distributed_rank <= 0:
        writer = SummaryWriter(args.save_dir)
        with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)
    else:
        writer = SummaryWriter(os.path.join(args.save_dir, str(args.distributed_rank)))

    # Train until the learning rate gets too small
    max_epoch = args.max_epoch or math.inf
    max_update = args.max_update or math.inf
    lr = trainer.get_lr()
    train_meter = StopwatchMeter()
    train_meter.start()
    while lr > args.min_lr and epoch <= max_epoch:

        if args.distributed_rank <= 0:
            writer.add_scalar('hyper/lr', lr, epoch)
            for form in ['QueKey', 'OutVal']:
                frobnorm, nucnorm, bound, nonorth = [], [], [], []
                for module in model.modules():
                    if hasattr(module, form.lower()):
                        U, VT = getattr(module, form.lower()).get_UVT()
                        for u, vt in zip(U, VT):
                            frobnorm.append(frobenius_norm(u, vt))
                            nucnorm.append(torch.norm(torch.matmul(u, vt), 'nuc'))
                            bound.append((u.pow(2).sum()+vt.pow(2).sum()) / 2.)
                            nonorth.append(sum(non_orthogonality(u, vt)) / 2.)
                writer.add_scalar('FrobNorm/'+form, sum(frobnorm) / len(frobnorm), epoch)
                writer.add_scalar('NucNorm/'+form, sum(nucnorm) / len(nucnorm), epoch)
                writer.add_scalar('NucNorm/'+form+'-Bound', sum(bound) / len(bound), epoch)
                writer.add_scalar('NonOrth/'+form, sum(nonorth) / len(nonorth), epoch)
            frobnorm, nucnorm, bound, nonorth = [], [], [], []
            for name, module in model.named_modules():
                if not any(block in name for block in ['embed', '_query', '_key', '_value', 'output_perform']):
                    if hasattr(module, 'frobgrad') and not hasattr(module, 'get_UVT'):
                        U, VT = module.U.data, module.VT.data
                        frobnorm.append(frobenius_norm(U, VT))
                        nucnorm.append(torch.norm(torch.matmul(U, VT), 'nuc'))
                        nonorth.append(sum(non_orthogonality(U, VT)) / 2.)
                        bound.append((U.pow(2).sum()+VT.pow(2).sum()) / 2.)
                    elif hasattr(module, 'weight'):
                        frobnorm.append(torch.norm(module.weight.data))
                        nucnorm.append(torch.norm(module.weight.data, 'nuc'))
            writer.add_scalar('FrobNorm/Linear', sum(frobnorm) / len(frobnorm), epoch)
            writer.add_scalar('NucNorm/Linear', sum(nucnorm) / len(nucnorm), epoch)
            if nonorth: 
                writer.add_scalar('NucNorm/Linear-Bound', sum(bound) / len(bound), epoch)
                writer.add_scalar('NonOrth/Linear', sum(nonorth) / len(nonorth), epoch)

        # train for one epoch
        train(args, trainer, dataset, epoch, batch_offset)

        # evaluate on validate set
        if epoch % args.validate_interval == 0:
            for k, subset in enumerate(args.valid_subset.split(',')):
                val_loss = validate(args, trainer, dataset, subset, epoch)
                if k == 0:
                    # only use first validation loss to update the learning schedule
                    lr = trainer.lr_step(epoch, val_loss)

                    # save checkpoint
                    if not args.no_save:
                        save_checkpoint(trainer, args, epoch, 0, val_loss)
            for k in ['loss', 'nll_loss']:
                writer.add_scalar('valid/'+k, trainer.meters['valid_'+k].avg, epoch)
                writer.add_scalar('train/'+k, trainer.meters['train_'+k].avg, epoch)
        else:
            lr = trainer.lr_step(epoch)

        epoch += 1
        batch_offset = 0

        if trainer.get_num_updates() >= max_update:
            break
    train_meter.stop()

    print('| done training in {:.1f} seconds'.format(train_meter.sum))
    writer.flush()
    newpar = sum(p.numel() for p in model.parameters())
    if 0.0 < args.rank_scale < 1.0:
        args.rank_scale = 1.0
        origpar = sum(p.numel() for p in models.build_model(args, dataset.src_dict, dataset.dst_dict).parameters())
    else:
        origpar = newpar
    if args.distributed_rank <= 0:
        with open(os.path.join(args.save_dir, 'results.json'), 'w') as f:
            json.dump({'final validation loss': trainer.meters['valid_nll_loss'].avg,
                       'original parameter count': origpar,
                       'compressed parameter count': newpar,
                       'compression ratio': newpar / origpar},
                       f, indent=4)


def train(args, trainer, dataset, epoch, batch_offset):
    """Train the model for one epoch."""

    # Set seed based on args.seed and the epoch number so that we get
    # reproducible results when resuming from checkpoints
    seed = args.seed + epoch
    torch.manual_seed(seed)

    # The max number of positions can be different for train and valid
    # e.g., RNNs may support more positions at test time than seen in training
    max_positions_train = (
        min(args.max_source_positions, trainer.get_model().max_encoder_positions()),
        min(args.max_target_positions, trainer.get_model().max_decoder_positions())
    )

    # Initialize dataloader, starting at batch_offset
    itr = dataset.train_dataloader(
        args.train_subset,
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions_train,
        seed=seed,
        epoch=epoch,
        sample_without_replacement=args.sample_without_replacement,
        sort_by_source_size=(epoch <= args.curriculum),
        shard_id=args.distributed_rank,
        num_shards=args.distributed_world_size,
    )
    progress = progress_bar.build_progress_bar(args, itr, epoch, no_progress_bar='simple')
    itr = itertools.islice(progress, batch_offset, None)

    # reset training meters
    for k in ['train_loss', 'train_nll_loss', 'wps', 'ups', 'wpb', 'bsz', 'clip']:
        meter = trainer.get_meter(k)
        if meter is not None:
            meter.reset()

    extra_meters = collections.defaultdict(lambda: AverageMeter())
    max_update = args.max_update or math.inf
    for i, sample in enumerate(itr, start=batch_offset):
        log_output = trainer.train_step(sample)

        # log mid-epoch stats
        stats = get_training_stats(trainer)
        for k, v in log_output.items():
            if k in ['loss', 'nll_loss']:
                continue  # these are already logged above
            if 'loss' in k:
                extra_meters[k].update(v, log_output['sample_size'])
            else:
                extra_meters[k].update(v)
            stats[k] = extra_meters[k].avg
        progress.log(stats)

        # save mid-epoch checkpoints
        if i == batch_offset:
            # ignore the first mini-batch in words-per-second calculation
            trainer.get_meter('wps').reset()

        # save mid-epoch checkpoints
        num_updates = trainer.get_num_updates()
        if args.save_interval > 0 and num_updates > 0 and num_updates % args.save_interval == 0:
            save_checkpoint(trainer, args, epoch, i + 1)

        if num_updates >= max_update:
            break

    # log end-of-epoch stats
    stats = get_training_stats(trainer)
    for k, meter in extra_meters.items():
        stats[k] = meter.avg
    progress.print(stats)


def get_training_stats(trainer):
    stats = collections.OrderedDict()
    stats['loss'] = '{:.3f}'.format(trainer.get_meter('train_loss').avg)
    if trainer.get_meter('train_nll_loss').count > 0:
        nll_loss = trainer.get_meter('train_nll_loss').avg
        stats['nll_loss'] = '{:.3f}'.format(nll_loss)
    else:
        nll_loss = trainer.get_meter('train_loss').avg
    stats['ppl'] = get_perplexity(nll_loss)
    stats['wps'] = round(trainer.get_meter('wps').avg)
    stats['ups'] = '{:.1f}'.format(trainer.get_meter('ups').avg)
    stats['wpb'] = round(trainer.get_meter('wpb').avg)
    stats['bsz'] = round(trainer.get_meter('bsz').avg)
    stats['num_updates'] = trainer.get_num_updates()
    stats['lr'] = trainer.get_lr()
    stats['gnorm'] = '{:.3f}'.format(trainer.get_meter('gnorm').avg)
    stats['clip'] = '{:.0%}'.format(trainer.get_meter('clip').avg)
    stats['oom'] = trainer.get_meter('oom').avg
    return stats


def validate(args, trainer, dataset, subset, epoch):
    """Evaluate the model on the validation set and return the average loss."""

    # Initialize dataloader
    max_positions_valid = (
        trainer.get_model().max_encoder_positions(),
        trainer.get_model().max_decoder_positions(),
    )
    itr = dataset.eval_dataloader(
        subset,
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences_valid,
        max_positions=max_positions_valid,
        skip_invalid_size_inputs_valid_test=args.skip_invalid_size_inputs_valid_test,
        descending=True,  # largest batch first to warm the caching allocator
        shard_id=args.distributed_rank,
        num_shards=args.distributed_world_size,
    )
    progress = progress_bar.build_progress_bar(
        args, itr, epoch,
        prefix='valid on \'{}\' subset'.format(subset),
        no_progress_bar='simple'
    )

    # reset validation loss meters
    for k in ['valid_loss', 'valid_nll_loss']:
        meter = trainer.get_meter(k)
        if meter is not None:
            meter.reset()

    extra_meters = collections.defaultdict(lambda: AverageMeter())
    for sample in progress:
        log_output = trainer.valid_step(sample)

        # log mid-validation stats
        stats = get_valid_stats(trainer)
        for k, v in log_output.items():
            if k in ['loss', 'nll_loss']:
                continue
            extra_meters[k].update(v)
            stats[k] = extra_meters[k].avg
        progress.log(stats)

    # log validation stats
    stats = get_valid_stats(trainer)
    for k, meter in extra_meters.items():
        stats[k] = meter.avg
    progress.print(stats)

    return stats['valid_loss']


def get_valid_stats(trainer):
    stats = collections.OrderedDict()
    stats['valid_loss'] = trainer.get_meter('valid_loss').avg
    if trainer.get_meter('valid_nll_loss').count > 0:
        nll_loss = trainer.get_meter('valid_nll_loss').avg
        stats['valid_nll_loss'] = nll_loss
    else:
        nll_loss = trainer.get_meter('valid_loss').avg
    stats['valid_ppl'] = get_perplexity(nll_loss)
    return stats


def get_perplexity(loss):
    try:
        return '{:.2f}'.format(math.pow(2, loss))
    except OverflowError:
        return float('inf')


def save_checkpoint(trainer, args, epoch, batch_offset, val_loss=None):
    extra_state = {
        'epoch': epoch,
        'batch_offset': batch_offset,
        'val_loss': val_loss,
    }

    if batch_offset == 0:
        if not args.no_epoch_checkpoints:
            epoch_filename = os.path.join(args.save_dir, 'checkpoint{}.pt'.format(epoch))
            trainer.save_checkpoint(epoch_filename, extra_state)

        assert val_loss is not None
        if not hasattr(save_checkpoint, 'best') or val_loss < save_checkpoint.best:
            save_checkpoint.best = val_loss
            best_filename = os.path.join(args.save_dir, 'checkpoint_best.pt')
            trainer.save_checkpoint(best_filename, extra_state)
    elif not args.no_epoch_checkpoints:
        epoch_filename = os.path.join(
            args.save_dir, 'checkpoint{}_{}.pt'.format(epoch, batch_offset))
        trainer.save_checkpoint(epoch_filename, extra_state)

    last_filename = os.path.join(args.save_dir, 'checkpoint_last.pt')
    trainer.save_checkpoint(last_filename, extra_state)


if __name__ == '__main__':
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser)
    main(args)
