import argparse
import multiprocessing
import os
import sys
import time
from math import floor
from subprocess import check_output
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import ujson
from torch.utils import data

from bootleg.trainer import Trainer
from bootleg.symbols.entity_symbols import EntitySymbols
from bootleg.symbols.constants import *
from bootleg.utils import utils, logging_utils, data_utils, train_utils, eval_utils
from bootleg.utils.parser_utils import get_full_config
from bootleg.utils.classes.dataset_collection import DatasetCollection
from bootleg.utils.classes.status_reporter import StatusReporter

def main(args, mode):
    multiprocessing.set_start_method("forkserver", force=True)
    # =================================
    # ARGUMENTS CHECK
    # =================================
    # distributed training
    assert (args.run_config.ngpus_per_node <= torch.cuda.device_count()) or (not torch.cuda.is_available()), 'Not enough GPUs per node.'
    world_size = args.run_config.ngpus_per_node * args.run_config.nodes
    if world_size > 1:
        args.run_config.distributed = True
    assert (args.run_config.distributed and world_size > 1) or (world_size == 1)

    train_utils.setup_run_folders(args, mode)

    # check slice method
    assert args.train_config.slice_method in SLICE_METHODS, f"You're slice_method {args.train_config.slice_method} is not in {SLICE_METHODS}."
    train_utils.setup_train_heads_and_eval_slices(args)

    # check save step
    assert args.run_config.save_every_k_eval > 0, f"You must have save_every_k_eval set to be > 0"

    # since eval, make sure resume model file is set and exists
    if mode == "eval" or mode == "dump_preds" or mode == "dump_embs":
        assert args.run_config.init_checkpoint != "", \
            f"You must specify a model checkpoint in run_config to run {mode}"
        assert os.path.exists(args.run_config.init_checkpoint),\
            f"The resume model file of {args.run_config.init_checkpoint} doesn't exist"

    utils.dump_json_file(filename=os.path.join(train_utils.get_save_folder(args.run_config), f"config_{mode}.json"), contents=args)
    if args.run_config.distributed:
        mp.spawn(main_worker, nprocs=args.run_config.ngpus_per_node, args=(args, mode, world_size))
    else:
        main_worker(gpu=args.run_config.gpu, args=args, mode=mode, world_size=world_size)


def main_worker(gpu, args, mode, world_size):
    # Explicitly setting seed to make sure that models created in two processes
    # start from same random weights and biases.
    torch.manual_seed(args.train_config.seed)
    np.random.seed(args.train_config.seed)
    # reset gpu if in distributed
    args.run_config.gpu = gpu
    logger = logging_utils.create_logger(args, mode)
    # dump log info before we change the batch size or else it's confusing why it doesn't
    # match the config settings
    logger.info(ujson.dumps(args, indent=4))
    logger.info("Machine: " + os.uname()[1])
    logger.info("CMD: python " + " ".join(sys.argv))
    # Dump git commit
    try:
        h = check_output(['git', 'rev-parse', '--short', 'HEAD']).strip()
        logger.info("Git hash: " + h.decode("utf-8"))
    except:
        logger.info("Git hash: git not found")

    is_writer = True
    rank = 0
    if args.run_config.distributed:
        # get process identifier number (based on node id and gpu id within a node)
        rank = args.run_config.nr * args.run_config.ngpus_per_node + args.run_config.gpu
        # this is blocking
        dist.init_process_group(
            backend='gloo',
            init_method=args.run_config.dist_url,
            world_size=world_size,
            rank=rank
        )
        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs we have
        args.train_config.batch_size = int(args.train_config.batch_size / args.run_config.ngpus_per_node)
        args.run_config.eval_batch_size = int(args.run_config.eval_batch_size / args.run_config.ngpus_per_node)
        # whether the machine is responsible for removing/writing files on the CPU
        is_writer = (rank % args.run_config.ngpus_per_node) == 0
    if torch.cuda.is_available() and not args.run_config.cpu:
        torch.cuda.set_device(args.run_config.gpu)
    if mode == 'train':
        train(args, is_writer=is_writer, logger=logger, world_size=world_size, rank=rank)
    elif mode == 'eval' or mode == 'dump_preds' or mode == 'dump_embs':
        model_eval(args, mode=mode, is_writer=is_writer, logger=logger,
                   world_size=world_size, rank=rank)
    else:
        raise Exception('--mode is either train or eval or fast_eval')


def train(args, is_writer, logger, world_size, rank):
    # This is main but call again in case train is called directly
    train_utils.setup_train_heads_and_eval_slices(args)
    train_utils.setup_run_folders(args, "train")

    # Load word symbols (like tokenizers) and entity symbols (aka entity profiles)
    word_symbols = data_utils.load_wordsymbols(args.data_config, is_writer, distributed=args.run_config.distributed)
    logger.info(f"Loading entity_symbols...")
    entity_symbols = EntitySymbols(load_dir=os.path.join(args.data_config.entity_dir, args.data_config.entity_map_dir),
        alias_cand_map_file=args.data_config.alias_cand_map)
    logger.info(f"Loaded entity_symbols with {entity_symbols.num_entities} entities.")
    # Get train dataset
    train_slice_dataset = data_utils.create_slice_dataset(args, args.data_config.train_dataset, is_writer, dataset_is_eval=False)
    train_dataset = data_utils.create_dataset(args, args.data_config.train_dataset, is_writer,
                                                  word_symbols, entity_symbols,
                                                  slice_dataset=train_slice_dataset,
                                                  dataset_is_eval=False)
    train_dataloader, train_sampler = data_utils.create_dataloader(args, train_dataset,
                                                                   eval_slice_dataset=None, world_size=world_size, rank=rank,
                                                                   batch_size=args.train_config.batch_size)

    # Repeat for dev
    dev_dataset_collection = {}
    dev_slice_dataset = data_utils.create_slice_dataset(args, args.data_config.dev_dataset, is_writer, dataset_is_eval=True)
    dev_dataset = data_utils.create_dataset(args, args.data_config.dev_dataset, is_writer,
                                            word_symbols, entity_symbols,
                                            slice_dataset=dev_slice_dataset,
                                            dataset_is_eval=True)
    dev_dataloader, dev_sampler = data_utils.create_dataloader(args, dev_dataset,
                                                               eval_slice_dataset=dev_slice_dataset, batch_size=args.run_config.eval_batch_size)
    dataset_collection = DatasetCollection(args.data_config.dev_dataset, args.data_config.dev_dataset.file, dev_dataset, dev_dataloader, dev_slice_dataset, dev_sampler)
    dev_dataset_collection[args.data_config.dev_dataset.file] = dataset_collection

    eval_slice_names = args.run_config.eval_slices

    total_steps_per_epoch = len(train_dataloader)
    # Create trainer---model, optimizer, and scorer
    trainer = Trainer(args, entity_symbols, word_symbols,
                      total_steps_per_epoch=total_steps_per_epoch,
                      eval_slice_names=eval_slice_names,
                      resume_model_file=args.run_config.init_checkpoint)

    # Set up epochs and intervals for saving and evaluating
    max_epochs = int(args.run_config.max_epochs)
    eval_steps = int(args.run_config.eval_steps)
    log_steps = int(args.run_config.log_steps)
    save_steps = max(int(args.run_config.save_every_k_eval * eval_steps), 1)
    logger.info(f"Eval steps {eval_steps}, Log steps {log_steps}, Save steps {save_steps}, Total training examples per epoch {len(train_dataset)}")
    status_reporter = StatusReporter(args, logger, is_writer, max_epochs, total_steps_per_epoch, is_eval=False)
    global_step = 0
    for epoch in range(trainer.start_epoch, trainer.start_epoch + max_epochs):
        # this is to fix having to save/restore the RNG state for checkpointing
        torch.manual_seed(args.train_config.seed + epoch)
        np.random.seed(args.train_config.seed + epoch)
        if args.run_config.distributed:
            # for determinism across runs https://github.com/pytorch/examples/issues/501
            train_sampler.set_epoch(epoch)

        start_time_load = time.time()
        for i, batch in enumerate(train_dataloader):
            load_time = time.time() - start_time_load
            start_time = time.time()
            _, loss_pack, _, _ = trainer.update(batch)
            # Log progress
            if (global_step+1) % log_steps == 0:
                duration = time.time() - start_time
                status_reporter.step_status(epoch=epoch, step=global_step, loss_pack=loss_pack, time=duration, load_time=load_time,
                                            lr=trainer.get_lr())
            # Save model
            if (global_step+1) % save_steps == 0 and is_writer:
                logger.info("Saving model...")
                trainer.save(save_dir=train_utils.get_save_folder(args.run_config), epoch=epoch, step=global_step, step_in_batch=i, suffix=args.run_config.model_suffix)
            # Run evaluation
            if (global_step+1) % eval_steps == 0:
                eval_utils.run_eval_all_dev_sets(args, global_step, dev_dataset_collection, logger, status_reporter, trainer)
            if args.run_config.distributed:
                dist.barrier()
            global_step += 1
            # Time loading new batch
            start_time_load = time.time()
        ######### END OF EPOCH
        if is_writer:
            logger.info(f"Saving model end of epoch {epoch}...")
            trainer.save(save_dir=train_utils.get_save_folder(args.run_config), epoch=epoch, step=global_step, step_in_batch=i, end_of_epoch=True, suffix=args.run_config.model_suffix)
        # Always run eval when saving -- if this coincided with eval_step, then don't need to rerun eval
        if (global_step+1) % eval_steps != 0:
            eval_utils.run_eval_all_dev_sets(args, global_step, dev_dataset_collection, logger, status_reporter, trainer)
    if is_writer:
        logger.info("Saving model...")
        trainer.save(save_dir=train_utils.get_save_folder(args.run_config), epoch=epoch, step=global_step, step_in_batch=i, end_of_epoch=True, last_epoch=True, suffix=args.run_config.model_suffix)
    if args.run_config.distributed:
        dist.barrier()


def model_eval(args, mode, is_writer, logger, world_size=1, rank=0):
    assert args.run_config.init_checkpoint != "", "You can't have an empty model file to do eval"
    # this is in main but call again in case eval is called directly
    train_utils.setup_train_heads_and_eval_slices(args)
    train_utils.setup_run_folders(args, mode)

    word_symbols = data_utils.load_wordsymbols(args.data_config, is_writer, distributed=args.run_config.distributed)
    logger.info(f"Loading entity_symbols...")
    entity_symbols = EntitySymbols(load_dir=os.path.join(args.data_config.entity_dir, args.data_config.entity_map_dir),
        alias_cand_map_file=args.data_config.alias_cand_map)
    logger.info(f"Loaded entity_symbols with {entity_symbols.num_entities} entities.")
    eval_slice_names = args.run_config.eval_slices
    test_dataset_collection = {}
    test_slice_dataset = data_utils.create_slice_dataset(args, args.data_config.test_dataset, is_writer, dataset_is_eval=True)
    test_dataset = data_utils.create_dataset(args, args.data_config.test_dataset, is_writer,
                                             word_symbols, entity_symbols,
                                             slice_dataset=test_slice_dataset,
                                             dataset_is_eval=True)
    test_dataloader, test_sampler = data_utils.create_dataloader(args, test_dataset,
                                                                 eval_slice_dataset=test_slice_dataset,
                                                                 batch_size=args.run_config.eval_batch_size)
    dataset_collection = DatasetCollection(args.data_config.test_dataset, args.data_config.test_dataset.file, test_dataset, test_dataloader, test_slice_dataset,
                                           test_sampler)
    test_dataset_collection[args.data_config.test_dataset.file] = dataset_collection

    trainer = Trainer(args, entity_symbols, word_symbols,
                      resume_model_file=args.run_config.init_checkpoint,
                      eval_slice_names=eval_slice_names,
                      model_eval=True)

    # Run evaluation numbers without dumping predictions (quick, batched)
    if mode == 'eval':
        status_reporter = StatusReporter(args, logger, is_writer, max_epochs=None, total_steps_per_epoch=None, is_eval=True)
        # results are written to json file
        for test_data_file in test_dataset_collection:
            logger.info(f"************************RUNNING EVAL {test_data_file}************************")
            test_dataloader = test_dataset_collection[test_data_file].data_loader
            # True is for if the batch is test or not, None is for the global step
            eval_utils.run_batched_eval(args=args,
                is_test=True, global_step=None, logger=logger, trainer=trainer, dataloader=test_dataloader,
                status_reporter=status_reporter, file=test_data_file)

    elif mode == 'dump_preds' or mode == 'dump_embs':
        # get predictions and optionally dump the corresponding contextual entity embeddings
        # TODO: support dumping ids for other embeddings as well (static entity embeddings, type embeddings, relation embeddings)
        # TODO: remove collection abstraction
        for test_data_file in test_dataset_collection:
            logger.info(f"************************DUMPING PREDICTIONS FOR {test_data_file}************************")
            test_dataloader = test_dataset_collection[test_data_file].data_loader
            pred_file, emb_file = eval_utils.run_dump_preds(args=args, entity_symbols=entity_symbols, test_data_file=test_data_file,
                logger=logger, trainer=trainer, dataloader=test_dataloader, dump_embs=(mode == 'dump_embs'))
            return pred_file, emb_file
    return


if __name__ == '__main__':
    config_parser = argparse.ArgumentParser('Where is config script?')
    config_parser.add_argument('--config_script', type=str, default='run_config.json',
                               help='This config should mimc the config.py config json with parameters you want to override.'
                                    'You can also override the parameters from config_script by passing them in directly after config_script. E.g., --train_config.batch_size 5')
    config_parser.add_argument('--mode', type=str, default='train', choices=["train", "eval", "dump_preds", "dump_embs"])
    # you can add other args that will override those in the config_script

    # parse_known_args returns 'args' that are the same as what parse_args() returns
    # and 'unknown' which are args that the parser doesn't recognize but you want to keep.
    # 'unknown' are what we pass on to our override any args from the second phase of arg parsing from the json config file
    args, unknown = config_parser.parse_known_args()
    final_args = get_full_config(args.config_script, unknown)
    main(final_args, args.mode)
