"""Trainer"""
from collections import OrderedDict
import copy
import numpy as np
import os
import shutil
import sys
import time
import torch
from torch import nn

from bootleg.model import Model
from bootleg.scorer import Scorer
from bootleg.optimizers.sparsedenseadam import SparseDenseAdam
from bootleg.eval_wrapper import EvalWrapper
from bootleg.symbols.constants import DISAMBIG, INDICATOR, FINAL_LOSS, BASE_SLICE
from bootleg.utils import logging_utils, model_utils, train_utils
from bootleg.utils.model_utils import count_parameters
from bootleg.utils.utils import import_class

class Trainer:
    """
    Trainer class: handles model loading, saving, and calling model.forward().
    The model loaded is declared in the config under model_config.base_model_load_class.
    """
    def __init__(self, args=None, entity_symbols=None, word_symbols=None,
        total_steps_per_epoch=0, resume_model_file="", eval_slice_names=None, model_eval=False):
        self.model_eval = model_eval # keep track of mode for model loading
        self.distributed = args.run_config.distributed
        self.args = args
        self.total_steps_per_epoch = total_steps_per_epoch
        self.start_epoch = 0
        self.start_step = 0
        self.use_cuda = not args.run_config.cpu and torch.cuda.is_available()
        self.logger = logging_utils.get_logger(args)
        if not self.use_cuda:
            self.model_device = "cpu"
            self.embedding_device = "cpu"
        else:
            self.model_device = args.run_config.gpu
            self.embedding_device = args.run_config.gpu
        # Load base model
        mod, load_class = import_class("bootleg", args.model_config.base_model_load_class)
        self.model = getattr(mod, load_class)(args=args, model_device=self.model_device, entity_symbols=entity_symbols, word_symbols=word_symbols)
        self.use_eval_wrapper = False
        if eval_slice_names is not None:
            self.use_eval_wrapper = True
            # Mapping of all output heads to indexes for the buffers
            head_key_to_idx = train_utils.get_head_key_to_idx(args)
            self.eval_wrapper = EvalWrapper(args=args, head_key_to_idx=head_key_to_idx, eval_slice_names=eval_slice_names,
                                            train_head_names=args.train_config.train_heads)
            self.eval_wrapper.to(self.model_device)
        self.optimizer = SparseDenseAdam(list(self.model.parameters()), lr=args.train_config.lr, weight_decay=args.train_config.weight_decay)

        self.scorer = Scorer(args, self.model_device)

        self.model.to(self.model_device)
        if self.distributed:
            # move everything to GPU
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.model_device], find_unused_parameters=True)

        # load model into existing model if model_file is provided
        if resume_model_file.endswith(".pt"):
            self.logger.info(f'Loading model from {resume_model_file}...')
            self.load(resume_model_file)

        self.logger.debug("Model device " + str(self.model_device))
        self.logger.debug("Embedding device " + str(self.embedding_device))
        self.logger.debug(f"*************************MODEL PARAMS WITH GRAD*************************")
        self.logger.debug(f'Number of model parameters with grad: {count_parameters(self.model, True, self.logger)}')
        self.logger.debug(f"*************************MODEL PARAMS WITHOUT GRAD*************************")
        self.logger.debug(f'Number of model parameters without grad: {count_parameters(self.model, False, self.logger)}')

    def get_lr(self):
        return model_utils.get_lr(self.optimizer)

    def update(self, batch, eval=False):
        if eval:
            torch.set_grad_enabled(False)
            self.model.eval()
        else:
            torch.set_grad_enabled(True)
            self.model.train()
            self.optimizer.zero_grad()
        # merge slice head labels for scorer
        true_entity_class = {DISAMBIG: {}, INDICATOR: {}}
        for head_name in self.args.train_config.train_heads:
            if head_name != BASE_SLICE or train_utils.model_has_base_head_loss(self.args):
                # These are labels for the correct disambiguation for a slice head. The labels are the same as for
                # final_loss except with examples not in the slice being masked out by -1
                true_entity_class[DISAMBIG][train_utils.get_slice_head_pred_name(head_name)] =\
                    batch[train_utils.get_slice_head_pred_name(head_name)].to(self.model_device)
                # Indicator labels for whether a mention is in the slice or not
                true_entity_class[INDICATOR][train_utils.get_slice_head_ind_name(head_name)] =\
                    batch[train_utils.get_slice_head_ind_name(head_name)].to(self.model_device)
        true_entity_class[DISAMBIG][FINAL_LOSS] = batch[train_utils.get_slice_head_pred_name(FINAL_LOSS)].to(self.model_device)

        # true_entity_class
        entity_indices = batch['entity_indices'].to(self.model_device)

        # Iterate over preprocessed embedding list and combine them into batch_prep arg. See wiki_dataset class for detailed comment on batch_prep.
        batch_prepped_data = {}
        for emb in self.args.data_config.ent_embeddings:
            if 'batch_prep' in emb and emb['batch_prep']:
                batch_prepped_data[emb.key] = batch[emb.key].to(self.model_device)
        # Iterate over embeddings that were prepped on the fly in the dataset class and add to batch_on_the_fly arg.
        batch_on_the_fly_data = {}
        for emb in self.args.data_config.ent_embeddings:
            if 'batch_on_the_fly' in emb and emb['batch_on_the_fly']:
                batch_on_the_fly_data[emb.key] = batch[emb.key].to(self.model_device)
        # Model forward pass.
        outs, entity_pack, final_entity_embs = self.model(
            alias_idx_pair_sent=[batch['start_idx_in_sent'].to(self.model_device),batch['end_idx_in_sent'].to(self.model_device)],
            word_indices=batch['word_indices'].to(self.model_device),
            alias_indices=batch['alias_idx'].to(self.model_device),
            entity_indices=entity_indices,
            batch_prepped_data=batch_prepped_data,
            batch_on_the_fly_data=batch_on_the_fly_data)
        loss_pack = self.scorer.calc_loss(outs, true_entity_class, entity_pack)
        # If using eval_wrapper
        if eval and self.use_eval_wrapper:
            self.eval_wrapper(
                slice_indices=batch['slice_indices'].to(self.model_device),
                true_label=true_entity_class,
                entity_indices=entity_indices,
                model_outs=outs)
            return outs, loss_pack, entity_pack, final_entity_embs
        # Else if doing analyze or more detailed eval
        if eval:
            return outs, loss_pack, entity_pack, final_entity_embs
        loss_pack.loss.backward()
        self.optimizer.step()
        return outs, loss_pack, entity_pack, final_entity_embs

    def save(self, save_dir, epoch, step, step_in_batch, end_of_epoch=False, last_epoch=False, suffix=""):
        model_state = self.model.state_dict()
        # Because of how our runs are saved, the only way to load a saved model is if you use the same model
        # args. These are saved in the run folder. We do not need to save the full set here.
        # Also, some model independent args, like cuda device, we want to be set at runtime, not from saved state.
        params = {
                'model': model_state,
                'epoch': epoch,
                # the step is dependent on the batch size so we save the batch size here
                'step': step_in_batch,
                'batch_size': self.args.train_config.batch_size,
                'optimizer': self.optimizer.state_dict()
                 }
        # If end_of_epoch, do not add step_str
        if end_of_epoch:
            step_str = ""
        else:
            step_str = f"_{step}"
        try:
            save_file = os.path.join(save_dir, f'model{epoch}{step_str}{suffix}.pt')
            torch.save(params, save_file)
            if last_epoch:
                last_file = os.path.join(save_dir, f'model{suffix}.pt')
                shutil.copyfile(save_file, last_file)
            self.logger.info("Model saved to {}".format(save_file))
        except Exception as e:
            self.logger.info("[Warning: Saving failed... continuing anyway.]")
            self.logger.info(e)

    def load(self, filename):
        try:
            if self.distributed:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(self.model_device)
                checkpoint = torch.load(filename, map_location=loc)
            else:
                checkpoint = torch.load(filename, lambda storage, loc: storage)
        except BaseException as e:
            self.logger.info("Cannot load model from {}".format(filename))
            self.logger.error(e)
            sys.exit(1)

        model_state_dict = checkpoint['model']

        if not self.distributed:
            # Remove distributed naming if model trained in distributed mode
            new_state_dict = OrderedDict()
            for k, v in model_state_dict.items():
                if k.startswith('module.'):
                    name = k[len('module.'):]
                    new_state_dict[name] = v
                else:
                    new_state_dict[k] = v
            model_state_dict = new_state_dict
        else:
            # If trained in non-distributed mode, rename to be distributed
            new_state_dict = OrderedDict()
            for k, v in model_state_dict.items():
                if not k.startswith('module.'):
                    name = f'module.{k}'
                    new_state_dict[name] = v
                else:
                    new_state_dict[k] = v
            model_state_dict = new_state_dict

        if not self.model_eval and self.args.train_config.load_optimizer_from_ckpt:
            if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.logger.info('Loaded optimizer from checkpoint.')
            else:
                self.logger.info('Optimizer not found in checkpoint. Skipping loading optimizer...')

        # If you load a DDP model into a normal model, the parameters won't match!!!! (strict=True will catch this)
        self.model.load_state_dict(model_state_dict, strict=True)

        if 'step' in checkpoint and not self.model_eval:
            self.start_step = checkpoint['step'] + 1
            # Roll over epoch
            if self.start_step >= self.total_steps_per_epoch:
                assert self.start_step == self.total_steps_per_epoch, 'The steps do not match total steps'
                self.start_step = 0
                self.start_epoch = checkpoint['epoch'] + 1
            else:
                raise ValueError('Continuing training from a partial epoch is not currently supported.')

            # Continuing training
            assert self.args.train_config.batch_size == checkpoint['batch_size'], 'The training batch size does not match that used for the ckpt. Cannot resume training from step.'
        else:
            self.start_epoch = checkpoint['epoch'] + 1
        self.logger.info(f"Successfully loaded model from {filename} starting from checkpoint epoch {self.start_epoch} and step {self.start_step}.")