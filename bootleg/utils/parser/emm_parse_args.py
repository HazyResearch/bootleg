"""Overrides the Emmental parse_args."""
import argparse
from argparse import ArgumentParser
from typing import Any, Dict, Optional, Tuple

from emmental.utils.utils import nullable_float, nullable_int, nullable_string, str2bool, str2dict

from bootleg.utils.classes.dotted_dict import DottedDict, createBoolDottedDict


def parse_args(parser: Optional[ArgumentParser] = None) -> Tuple[ArgumentParser, Dict]:
    """Parser. Overrides the default Emmental parser to add the "emmental."
    level to the parser so we can parse it correctly with the Bootleg config.

    Args:
      parser: Argument parser object, defaults to None.

    Returns:
      The updated argument parser object.
    """
    if parser is None:
        parser = argparse.ArgumentParser(
            "Emmental configuration",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    parser_hierarchy = {"emmental": {}}
    # Load meta configuration
    meta_config = parser.add_argument_group("Meta configuration")

    meta_config.add_argument(
        "--emmental.seed",
        type=nullable_int,
        default=1234,
        help="Random seed for all numpy/torch/cuda operations in model and learning",
    )

    meta_config.add_argument(
        "--emmental.verbose",
        type=str2bool,
        default=True,
        help="Whether to print the log information",
    )

    meta_config.add_argument(
        "--emmental.log_path",
        type=str,
        default="logs",
        help="Directory to save running log",
    )

    meta_config.add_argument(
        "--emmental.use_exact_log_path",
        type=str2bool,
        default=False,
        help="Whether to use the exact log directory",
    )
    parser_hierarchy["emmental"]["_global_meta"] = meta_config

    # Load data configuration
    data_config = parser.add_argument_group("Data configuration")

    data_config.add_argument(
        "--emmental.min_data_len", type=int, default=0, help="Minimal data length"
    )

    data_config.add_argument(
        "--emmental.max_data_len",
        type=int,
        default=0,
        help="Maximal data length (0 for no max_len)",
    )
    parser_hierarchy["emmental"]["_global_data"] = data_config

    # Load model configuration
    model_config = parser.add_argument_group("Model configuration")

    model_config.add_argument(
        "--emmental.model_path",
        type=nullable_string,
        default=None,
        help="Path to pretrained model",
    )

    model_config.add_argument(
        "--emmental.device",
        type=int,
        default=0,
        help="Which device to use (-1 for cpu or gpu id (e.g., 0 for cuda:0))",
    )

    model_config.add_argument(
        "--emmental.dataparallel",
        type=str2bool,
        default=False,
        help="Whether to use dataparallel or not",
    )

    model_config.add_argument(
        "--emmental.distributed_backend",
        type=str,
        default="nccl",
        choices=["nccl", "gloo"],
        help="Which backend to use for distributed training.",
    )

    parser_hierarchy["emmental"]["_global_model"] = model_config

    # Learning configuration
    learner_config = parser.add_argument_group("Learning configuration")

    learner_config.add_argument(
        "--emmental.optimizer_path",
        type=nullable_string,
        default=None,
        help="Path to optimizer state",
    )

    learner_config.add_argument(
        "--emmental.scheduler_path",
        type=nullable_string,
        default=None,
        help="Path to lr scheduler state",
    )

    learner_config.add_argument(
        "--emmental.fp16",
        type=str2bool,
        default=False,
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex)"
        "instead of 32-bit",
    )

    learner_config.add_argument(
        "--emmental.fp16_opt_level",
        type=str,
        default="O1",
        help="Apex AMP optimization level selected in ['O0', 'O1', 'O2', 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )

    learner_config.add_argument(
        "--emmental.local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus",
    )

    learner_config.add_argument(
        "--emmental.epochs_learned", type=int, default=0, help="Learning epochs learned"
    )

    learner_config.add_argument(
        "--emmental.n_epochs",
        type=int,
        default=1,
        help="Total number of learning epochs",
    )

    learner_config.add_argument(
        "--emmental.steps_learned", type=int, default=0, help="Learning steps learned"
    )

    learner_config.add_argument(
        "--emmental.n_steps",
        type=int,
        default=None,
        help="Total number of learning steps",
    )

    learner_config.add_argument(
        "--emmental.skip_learned_data",
        type=str2bool,
        default=False,
        help="Iterate through dataloader when steps or epochs learned is true",
    )

    learner_config.add_argument(
        "--emmental.train_split",
        nargs="+",
        type=str,
        default=["train"],
        help="The split for training",
    )

    learner_config.add_argument(
        "--emmental.valid_split",
        nargs="+",
        type=str,
        default=["dev"],
        help="The split for validation",
    )

    learner_config.add_argument(
        "--emmental.test_split",
        nargs="+",
        type=str,
        default=["test"],
        help="The split for testing",
    )

    learner_config.add_argument(
        "--emmental.ignore_index",
        type=nullable_int,
        default=None,
        help="The ignore index, uses for masking samples",
    )

    learner_config.add_argument(
        "--emmental.online_eval",
        type=str2bool,
        default=False,
        help="Whether to perform online evaluation",
    )

    parser_hierarchy["emmental"]["_global_learner"] = learner_config

    # Optimizer configuration
    optimizer_config = parser.add_argument_group("Optimizer configuration")

    optimizer_config.add_argument(
        "--emmental.optimizer",
        type=nullable_string,
        default="adamw",
        choices=[
            "asgd",
            "adadelta",
            "adagrad",
            "adam",
            "adamw",
            "adamax",
            "lbfgs",
            "rms_prop",
            "r_prop",
            "sgd",
            "sparse_adam",
            "bert_adam",
            None,
        ],
        help="The optimizer to use",
    )

    optimizer_config.add_argument(
        "--emmental.lr", type=float, default=1e-3, help="Learing rate"
    )

    optimizer_config.add_argument(
        "--emmental.l2", type=float, default=0.0, help="l2 regularization"
    )

    optimizer_config.add_argument(
        "--emmental.grad_clip",
        type=nullable_float,
        default=None,
        help="Gradient clipping",
    )

    optimizer_config.add_argument(
        "--emmental.gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )

    # ASGD config
    optimizer_config.add_argument(
        "--emmental.asgd_lambd", type=float, default=0.0001, help="ASGD lambd"
    )

    optimizer_config.add_argument(
        "--emmental.asgd_alpha", type=float, default=0.75, help="ASGD alpha"
    )

    optimizer_config.add_argument(
        "--emmental.asgd_t0", type=float, default=1000000.0, help="ASGD t0"
    )

    # Adadelta config
    optimizer_config.add_argument(
        "--emmental.adadelta_rho", type=float, default=0.9, help="Adadelta rho"
    )

    optimizer_config.add_argument(
        "--emmental.adadelta_eps", type=float, default=0.000001, help="Adadelta eps"
    )

    # Adagrad config
    optimizer_config.add_argument(
        "--emmental.adagrad_lr_decay", type=float, default=0, help="Adagrad lr_decay"
    )

    optimizer_config.add_argument(
        "--emmental.adagrad_initial_accumulator_value",
        type=float,
        default=0,
        help="Adagrad initial accumulator value",
    )

    optimizer_config.add_argument(
        "--emmental.adagrad_eps", type=float, default=0.0000000001, help="Adagrad eps"
    )

    # Adam config
    optimizer_config.add_argument(
        "--emmental.adam_betas",
        nargs="+",
        type=float,
        default=(0.9, 0.999),
        help="Adam betas",
    )

    optimizer_config.add_argument(
        "--emmental.adam_eps", type=float, default=1e-6, help="Adam eps"
    )

    optimizer_config.add_argument(
        "--emmental.adam_amsgrad",
        type=str2bool,
        default=False,
        help="Whether to use the AMSGrad variant of adam",
    )

    # AdamW config
    optimizer_config.add_argument(
        "--emmental.adamw_betas",
        nargs="+",
        type=float,
        default=(0.9, 0.999),
        help="AdamW betas",
    )

    optimizer_config.add_argument(
        "--emmental.adamw_eps", type=float, default=1e-6, help="AdamW eps"
    )

    optimizer_config.add_argument(
        "--emmental.adamw_amsgrad",
        type=str2bool,
        default=False,
        help="Whether to use the AMSGrad variant of AdamW",
    )

    # Adamax config
    optimizer_config.add_argument(
        "--emmental.adamax_betas",
        nargs="+",
        type=float,
        default=(0.9, 0.999),
        help="Adamax betas",
    )

    optimizer_config.add_argument(
        "--emmental.adamax_eps", type=float, default=1e-6, help="Adamax eps"
    )

    # LBFGS config
    optimizer_config.add_argument(
        "--emmental.lbfgs_max_iter", type=int, default=20, help="LBFGS max iter"
    )

    optimizer_config.add_argument(
        "--emmental.lbfgs_max_eval",
        type=nullable_int,
        default=None,
        help="LBFGS max eval",
    )

    optimizer_config.add_argument(
        "--emmental.lbfgs_tolerance_grad",
        type=float,
        default=1e-07,
        help="LBFGS tolerance grad",
    )

    optimizer_config.add_argument(
        "--emmental.lbfgs_tolerance_change",
        type=float,
        default=1e-09,
        help="LBFGS tolerance change",
    )

    optimizer_config.add_argument(
        "--emmental.lbfgs_history_size",
        type=int,
        default=100,
        help="LBFGS history size",
    )

    optimizer_config.add_argument(
        "--emmental.lbfgs_line_search_fn",
        type=nullable_string,
        default=None,
        help="LBFGS line search fn",
    )

    # RMSprop config
    optimizer_config.add_argument(
        "--emmental.rms_prop_alpha", type=float, default=0.99, help="RMSprop alpha"
    )

    optimizer_config.add_argument(
        "--emmental.rms_prop_eps", type=float, default=1e-08, help="RMSprop eps"
    )

    optimizer_config.add_argument(
        "--emmental.rms_prop_momentum", type=float, default=0, help="RMSprop momentum"
    )

    optimizer_config.add_argument(
        "--emmental.rms_prop_centered",
        type=str2bool,
        default=False,
        help="RMSprop centered",
    )

    # Rprop config
    optimizer_config.add_argument(
        "--emmental.r_prop_etas",
        nargs="+",
        type=float,
        default=(0.5, 1.2),
        help="Rprop etas",
    )

    optimizer_config.add_argument(
        "--emmental.r_prop_step_sizes",
        nargs="+",
        type=float,
        default=(1e-06, 50),
        help="Rprop step sizes",
    )

    # SGD config
    optimizer_config.add_argument(
        "--emmental.sgd_momentum", type=float, default=0, help="SGD momentum"
    )

    optimizer_config.add_argument(
        "--emmental.sgd_dampening", type=float, default=0, help="SGD dampening"
    )

    optimizer_config.add_argument(
        "--emmental.sgd_nesterov", type=str2bool, default=False, help="SGD nesterov"
    )

    # SparseAdam config
    optimizer_config.add_argument(
        "--emmental.sparse_adam_betas",
        nargs="+",
        type=float,
        default=(0.9, 0.999),
        help="SparseAdam betas",
    )

    optimizer_config.add_argument(
        "--emmental.sparse_adam_eps", type=float, default=1e-06, help="SparseAdam eps"
    )

    # BertAdam config
    optimizer_config.add_argument(
        "--emmental.bert_adam_betas",
        nargs="+",
        type=float,
        default=(0.9, 0.999),
        help="BertAdam betas",
    )

    optimizer_config.add_argument(
        "--emmental.bert_adam_eps", type=float, default=1e-06, help="BertAdam eps"
    )

    parser_hierarchy["emmental"]["_global_optimizer"] = optimizer_config

    # Scheduler configuration
    scheduler_config = parser.add_argument_group("Scheduler configuration")

    scheduler_config.add_argument(
        "--emmental.lr_scheduler",
        type=nullable_string,
        default=None,
        choices=[
            "linear",
            "exponential",
            "plateau",
            "step",
            "multi_step",
            "cyclic",
            "one_cycle",
            "cosine_annealing",
        ],
        help="Learning rate scheduler",
    )

    scheduler_config.add_argument(
        "--emmental.lr_scheduler_step_unit",
        type=str,
        default="batch",
        choices=["batch", "epoch"],
        help="Learning rate scheduler step unit",
    )

    scheduler_config.add_argument(
        "--emmental.lr_scheduler_step_freq",
        type=int,
        default=1,
        help="Learning rate scheduler step freq",
    )

    scheduler_config.add_argument(
        "--emmental.warmup_steps", type=float, default=None, help="Warm up steps"
    )

    scheduler_config.add_argument(
        "--emmental.warmup_unit",
        type=str,
        default="batch",
        choices=["batch", "epoch"],
        help="Warm up unit",
    )

    scheduler_config.add_argument(
        "--emmental.warmup_percentage",
        type=float,
        default=None,
        help="Warm up percentage",
    )

    scheduler_config.add_argument(
        "--emmental.min_lr", type=float, default=0.0, help="Minimum learning rate"
    )

    scheduler_config.add_argument(
        "--emmental.reset_state",
        type=str2bool,
        default=False,
        help="Whether reset the state of the optimizer when lr changes",
    )

    scheduler_config.add_argument(
        "--emmental.exponential_lr_scheduler_gamma",
        type=float,
        default=0.9,
        help="Gamma for exponential lr scheduler",
    )

    # ReduceLROnPlateau lr scheduler config
    scheduler_config.add_argument(
        "--emmental.plateau_lr_scheduler_metric",
        type=str,
        default="model/train/all/loss",
        help="Metric of plateau lr scheduler",
    )

    scheduler_config.add_argument(
        "--emmental.plateau_lr_scheduler_mode",
        type=str,
        default="min",
        choices=["min", "max"],
        help="Mode of plateau lr scheduler",
    )

    scheduler_config.add_argument(
        "--emmental.plateau_lr_scheduler_factor",
        type=float,
        default=0.1,
        help="Factor of plateau lr scheduler",
    )

    scheduler_config.add_argument(
        "--emmental.plateau_lr_scheduler_patience",
        type=int,
        default=10,
        help="Patience for plateau lr scheduler",
    )

    scheduler_config.add_argument(
        "--emmental.plateau_lr_scheduler_threshold",
        type=float,
        default=0.0001,
        help="Threshold of plateau lr scheduler",
    )

    scheduler_config.add_argument(
        "--emmental.plateau_lr_scheduler_threshold_mode",
        type=str,
        default="rel",
        choices=["rel", "abs"],
        help="Threshold mode of plateau lr scheduler",
    )

    scheduler_config.add_argument(
        "--emmental.plateau_lr_scheduler_cooldown",
        type=int,
        default=0,
        help="Cooldown of plateau lr scheduler",
    )

    scheduler_config.add_argument(
        "--emmental.plateau_lr_scheduler_eps",
        type=float,
        default=0.00000001,
        help="Eps of plateau lr scheduler",
    )

    # Step lr scheduler config
    scheduler_config.add_argument(
        "--emmental.step_lr_scheduler_step_size",
        type=int,
        default=1,
        help="Period of learning rate decay",
    )

    scheduler_config.add_argument(
        "--emmental.step_lr_scheduler_gamma",
        type=float,
        default=0.1,
        help="Multiplicative factor of learning rate decay",
    )

    scheduler_config.add_argument(
        "--emmental.step_lr_scheduler_last_epoch",
        type=int,
        default=-1,
        help="The index of last epoch",
    )

    scheduler_config.add_argument(
        "--emmental.multi_step_lr_scheduler_milestones",
        nargs="+",
        type=int,
        default=[1000],
        help="List of epoch indices. Must be increasing.",
    )

    scheduler_config.add_argument(
        "--emmental.multi_step_lr_scheduler_gamma",
        type=float,
        default=0.1,
        help="Multiplicative factor of learning rate decay",
    )

    scheduler_config.add_argument(
        "--emmental.multi_step_lr_scheduler_last_epoch",
        type=int,
        default=-1,
        help="The index of last epoch",
    )

    # Cyclic lr scheduler config
    scheduler_config.add_argument(
        "--emmental.cyclic_lr_scheduler_base_lr",
        nargs="+",
        type=float,
        default=0.001,
        help="Base lr of cyclic lr scheduler",
    )

    scheduler_config.add_argument(
        "--emmental.cyclic_lr_scheduler_max_lr",
        nargs="+",
        type=float,
        default=0.1,
        help="Max lr of cyclic lr scheduler",
    )

    scheduler_config.add_argument(
        "--emmental.cyclic_lr_scheduler_step_size_up",
        type=int,
        default=2000,
        help="Step size up of cyclic lr scheduler",
    )

    scheduler_config.add_argument(
        "--emmental.cyclic_lr_scheduler_step_size_down",
        type=nullable_int,
        default=None,
        help="Step size down of cyclic lr scheduler",
    )

    scheduler_config.add_argument(
        "--emmental.cyclic_lr_scheduler_mode",
        type=nullable_string,
        default="triangular",
        help="Mode of cyclic lr scheduler",
    )

    scheduler_config.add_argument(
        "--emmental.cyclic_lr_scheduler_gamma",
        type=float,
        default=1.0,
        help="Gamma of cyclic lr scheduler",
    )

    # TODO: support cyclic_lr_scheduler_scale_fn

    scheduler_config.add_argument(
        "--emmental.cyclic_lr_scheduler_scale_mode",
        type=str,
        default="cycle",
        choices=["cycle", "iterations"],
        help="Scale mode of cyclic lr scheduler",
    )

    scheduler_config.add_argument(
        "--emmental.cyclic_lr_scheduler_cycle_momentum",
        type=str2bool,
        default=True,
        help="Cycle momentum of cyclic lr scheduler",
    )

    scheduler_config.add_argument(
        "--emmental.cyclic_lr_scheduler_base_momentum",
        nargs="+",
        type=float,
        default=0.8,
        help="Base momentum of cyclic lr scheduler",
    )

    scheduler_config.add_argument(
        "--emmental.cyclic_lr_scheduler_max_momentum",
        nargs="+",
        type=float,
        default=0.9,
        help="Max momentum of cyclic lr scheduler",
    )

    scheduler_config.add_argument(
        "--emmental.cyclic_lr_scheduler_last_epoch",
        type=int,
        default=-1,
        help="Last epoch of cyclic lr scheduler",
    )

    # One cycle lr scheduler config
    scheduler_config.add_argument(
        "--emmental.one_cycle_lr_scheduler_max_lr",
        nargs="+",
        type=float,
        default=0.1,
        help="Max lr of one cyclic lr scheduler",
    )

    scheduler_config.add_argument(
        "--emmental.one_cycle_lr_scheduler_pct_start",
        type=float,
        default=0.3,
        help="Percentage start of one cyclic lr scheduler",
    )

    scheduler_config.add_argument(
        "--emmental.one_cycle_lr_scheduler_anneal_strategy",
        type=str,
        default="cos",
        choices=["cos", "linear"],
        help="Anneal strategyr of one cyclic lr scheduler",
    )

    scheduler_config.add_argument(
        "--emmental.one_cycle_lr_scheduler_cycle_momentum",
        type=str2bool,
        default=True,
        help="Cycle momentum of one cyclic lr scheduler",
    )

    scheduler_config.add_argument(
        "--emmental.one_cycle_lr_scheduler_base_momentum",
        nargs="+",
        type=float,
        default=0.85,
        help="Base momentum of one cyclic lr scheduler",
    )

    scheduler_config.add_argument(
        "--emmental.one_cycle_lr_scheduler_max_momentum",
        nargs="+",
        type=float,
        default=0.95,
        help="Max momentum of one cyclic lr scheduler",
    )

    scheduler_config.add_argument(
        "--emmental.one_cycle_lr_scheduler_div_factor",
        type=float,
        default=25,
        help="Div factor of one cyclic lr scheduler",
    )

    scheduler_config.add_argument(
        "--emmental.one_cycle_lr_scheduler_final_div_factor",
        type=float,
        default=1e4,
        help="Final div factor of one cyclic lr scheduler",
    )

    scheduler_config.add_argument(
        "--emmental.one_cycle_lr_scheduler_last_epoch",
        type=int,
        default=-1,
        help="Last epoch of one cyclic lr scheduler",
    )

    scheduler_config.add_argument(
        "--emmental.cosine_annealing_lr_scheduler_last_epoch",
        type=int,
        default=-1,
        help="The index of last epoch",
    )

    scheduler_config.add_argument(
        "--emmental.task_scheduler",
        type=str,
        default="round_robin",
        # choices=["sequential", "round_robin", "mixed"],
        help="Task scheduler",
    )

    scheduler_config.add_argument(
        "--emmental.sequential_scheduler_fillup",
        type=str2bool,
        default=False,
        help="Whether fillup in sequential scheduler",
    )

    scheduler_config.add_argument(
        "--emmental.round_robin_scheduler_fillup",
        type=str2bool,
        default=False,
        help="whether fillup in round robin scheduler",
    )

    scheduler_config.add_argument(
        "--emmental.mixed_scheduler_fillup",
        type=str2bool,
        default=False,
        help="whether fillup in mixed scheduler scheduler",
    )
    parser_hierarchy["emmental"]["_global_scheduler"] = scheduler_config

    # Logging configuration
    logging_config = parser.add_argument_group("Logging configuration")

    logging_config.add_argument(
        "--emmental.counter_unit",
        type=str,
        default="epoch",
        choices=["epoch", "batch"],
        help="Logging unit (epoch, batch)",
    )

    logging_config.add_argument(
        "--emmental.evaluation_freq",
        type=float,
        default=1,
        help="Logging evaluation frequency",
    )

    logging_config.add_argument(
        "--emmental.writer",
        type=str,
        default="tensorboard",
        choices=["json", "tensorboard", "wandb"],
        help="The writer format (json, tensorboard, wandb)",
    )

    logging_config.add_argument(
        "--emmental.write_loss_per_step",
        type=bool,
        default=False,
        help="Whether to log loss per step",
    )

    logging_config.add_argument(
        "--emmental.wandb_project_name",
        type=nullable_string,
        default=None,
        help="Wandb project name",
    )

    logging_config.add_argument(
        "--emmental.wandb_run_name",
        type=nullable_string,
        default=None,
        help="Wandb run name",
    )

    logging_config.add_argument(
        "--emmental.wandb_watch_model",
        type=bool,
        default=False,
        help="Whether use wandb to watch model",
    )

    logging_config.add_argument(
        "--emmental.wandb_model_watch_freq",
        type=nullable_int,
        default=None,
        help="Wandb model watch frequency",
    )

    logging_config.add_argument(
        "--emmental.checkpointing",
        type=str2bool,
        default=True,
        help="Whether to checkpoint the model",
    )

    logging_config.add_argument(
        "--emmental.checkpoint_path", type=str, default=None, help="Checkpointing path"
    )

    logging_config.add_argument(
        "--emmental.checkpoint_freq",
        type=int,
        default=1,
        help="Checkpointing every k logging time",
    )

    logging_config.add_argument(
        "--emmental.checkpoint_metric",
        type=str2dict,
        default={"model/train/all/loss": "min"},
        help=(
            "Checkpointing metric (metric_name:mode), "
            "e.g., `model/train/all/loss:min`"
        ),
    )

    logging_config.add_argument(
        "--emmental.checkpoint_task_metrics",
        type=str2dict,
        default=None,
        help=(
            "Task specific checkpointing metric "
            "(metric_name1:mode1,metric_name2:mode2)"
        ),
    )

    logging_config.add_argument(
        "--emmental.checkpoint_runway",
        type=float,
        default=0,
        help="Checkpointing runway (no checkpointing before k checkpointing unit)",
    )

    logging_config.add_argument(
        "--emmental.checkpoint_all",
        type=str2bool,
        default=True,
        help="Whether to checkpoint all checkpoints",
    )

    logging_config.add_argument(
        "--emmental.clear_intermediate_checkpoints",
        type=str2bool,
        default=False,
        help="Whether to clear intermediate checkpoints",
    )

    logging_config.add_argument(
        "--emmental.clear_all_checkpoints",
        type=str2bool,
        default=False,
        help="Whether to clear all checkpoints",
    )
    parser_hierarchy["emmental"]["_global_logging"] = logging_config
    return parser, parser_hierarchy


def parse_args_to_config(args: DottedDict) -> Dict[str, Any]:
    """Parse the Emmental arguments to config dict.

    Args:
      args: parsed namespace from argument parser.

    Returns: Emmental config dict.
    """
    config = {
        "meta_config": {
            "seed": args.seed,
            "verbose": args.verbose,
            "log_path": args.log_path,
            "use_exact_log_path": args.use_exact_log_path,
        },
        "data_config": {
            "min_data_len": args.min_data_len,
            "max_data_len": args.max_data_len,
        },
        "model_config": {
            "model_path": args.model_path,
            "device": args.device,
            "dataparallel": args.dataparallel,
            "distributed_backend": args.distributed_backend,
        },
        "learner_config": {
            "optimizer_path": args.optimizer_path,
            "scheduler_path": args.scheduler_path,
            "fp16": args.fp16,
            "fp16_opt_level": args.fp16_opt_level,
            "local_rank": args.local_rank,
            "epochs_learned": args.epochs_learned,
            "n_epochs": args.n_epochs,
            "steps_learned": args.steps_learned,
            "n_steps": args.n_steps,
            "skip_learned_data": args.skip_learned_data,
            "train_split": args.train_split,
            "valid_split": args.valid_split,
            "test_split": args.test_split,
            "ignore_index": args.ignore_index,
            "online_eval": args.online_eval,
            "optimizer_config": {
                "optimizer": args.optimizer,
                "lr": args.lr,
                "l2": args.l2,
                "grad_clip": args.grad_clip,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "asgd_config": {
                    "lambd": args.asgd_lambd,
                    "alpha": args.asgd_alpha,
                    "t0": args.asgd_t0,
                },
                "adadelta_config": {"rho": args.adadelta_rho, "eps": args.adadelta_eps},
                "adagrad_config": {
                    "lr_decay": args.adagrad_lr_decay,
                    "initial_accumulator_value": args.adagrad_initial_accumulator_value,
                    "eps": args.adagrad_eps,
                },
                "adam_config": {
                    "betas": args.adam_betas,
                    "amsgrad": args.adam_amsgrad,
                    "eps": args.adam_eps,
                },
                "adamw_config": {
                    "betas": args.adamw_betas,
                    "amsgrad": args.adamw_amsgrad,
                    "eps": args.adamw_eps,
                },
                "adamax_config": {"betas": args.adamax_betas, "eps": args.adamax_eps},
                "lbfgs_config": {
                    "max_iter": args.lbfgs_max_iter,
                    "max_eval": args.lbfgs_max_eval,
                    "tolerance_grad": args.lbfgs_tolerance_grad,
                    "tolerance_change": args.lbfgs_tolerance_change,
                    "history_size": args.lbfgs_history_size,
                    "line_search_fn": args.lbfgs_line_search_fn,
                },
                "rms_prop_config": {
                    "alpha": args.rms_prop_alpha,
                    "eps": args.rms_prop_eps,
                    "momentum": args.rms_prop_momentum,
                    "centered": args.rms_prop_centered,
                },
                "r_prop_config": {
                    "etas": args.r_prop_etas,
                    "step_sizes": args.r_prop_step_sizes,
                },
                "sgd_config": {
                    "momentum": args.sgd_momentum,
                    "dampening": args.sgd_dampening,
                    "nesterov": args.sgd_nesterov,
                },
                "sparse_adam_config": {
                    "betas": args.sparse_adam_betas,
                    "eps": args.sparse_adam_eps,
                },
                "bert_adam_config": {
                    "betas": args.bert_adam_betas,
                    "eps": args.bert_adam_eps,
                },
            },
            "lr_scheduler_config": {
                "lr_scheduler": args.lr_scheduler,
                "lr_scheduler_step_unit": args.lr_scheduler_step_unit,
                "lr_scheduler_step_freq": args.lr_scheduler_step_freq,
                "warmup_steps": args.warmup_steps,
                "warmup_unit": args.warmup_unit,
                "warmup_percentage": args.warmup_percentage,
                "min_lr": args.min_lr,
                "reset_state": args.reset_state,
                "exponential_config": {"gamma": args.exponential_lr_scheduler_gamma},
                "plateau_config": {
                    "metric": args.plateau_lr_scheduler_metric,
                    "mode": args.plateau_lr_scheduler_mode,
                    "factor": args.plateau_lr_scheduler_factor,
                    "patience": args.plateau_lr_scheduler_patience,
                    "threshold": args.plateau_lr_scheduler_threshold,
                    "threshold_mode": args.plateau_lr_scheduler_threshold_mode,
                    "cooldown": args.plateau_lr_scheduler_cooldown,
                    "eps": args.plateau_lr_scheduler_eps,
                },
                "step_config": {
                    "step_size": args.step_lr_scheduler_step_size,
                    "gamma": args.step_lr_scheduler_gamma,
                    "last_epoch": args.step_lr_scheduler_last_epoch,
                },
                "multi_step_config": {
                    "milestones": args.multi_step_lr_scheduler_milestones,
                    "gamma": args.multi_step_lr_scheduler_gamma,
                    "last_epoch": args.multi_step_lr_scheduler_last_epoch,
                },
                "cyclic_config": {
                    "base_lr": args.cyclic_lr_scheduler_base_lr,
                    "max_lr": args.cyclic_lr_scheduler_max_lr,
                    "step_size_up": args.cyclic_lr_scheduler_step_size_up,
                    "step_size_down": args.cyclic_lr_scheduler_step_size_down,
                    "mode": args.cyclic_lr_scheduler_mode,
                    "gamma": args.cyclic_lr_scheduler_gamma,
                    "scale_fn": None,
                    "scale_mode": args.cyclic_lr_scheduler_scale_mode,
                    "cycle_momentum": args.cyclic_lr_scheduler_cycle_momentum,
                    "base_momentum": args.cyclic_lr_scheduler_base_momentum,
                    "max_momentum": args.cyclic_lr_scheduler_max_momentum,
                    "last_epoch": args.cyclic_lr_scheduler_last_epoch,
                },
                "one_cycle_config": {
                    "max_lr": args.one_cycle_lr_scheduler_max_lr,
                    "pct_start": args.one_cycle_lr_scheduler_pct_start,
                    "anneal_strategy": args.one_cycle_lr_scheduler_anneal_strategy,
                    "cycle_momentum": args.one_cycle_lr_scheduler_cycle_momentum,
                    "base_momentum": args.one_cycle_lr_scheduler_base_momentum,
                    "max_momentum": args.one_cycle_lr_scheduler_max_momentum,
                    "div_factor": args.one_cycle_lr_scheduler_div_factor,
                    "final_div_factor": args.one_cycle_lr_scheduler_final_div_factor,
                    "last_epoch": args.one_cycle_lr_scheduler_last_epoch,
                },
                "cosine_annealing_config": {
                    "last_epoch": args.cosine_annealing_lr_scheduler_last_epoch
                },
            },
            "task_scheduler_config": {
                "task_scheduler": args.task_scheduler,
                "sequential_scheduler_config": {
                    "fillup": args.sequential_scheduler_fillup
                },
                "round_robin_scheduler_config": {
                    "fillup": args.round_robin_scheduler_fillup
                },
                "mixed_scheduler_config": {"fillup": args.mixed_scheduler_fillup},
            },
        },
        "logging_config": {
            "counter_unit": args.counter_unit,
            "evaluation_freq": args.evaluation_freq,
            "writer_config": {
                "verbose": True,
                "writer": args.writer,
                "write_loss_per_step": args.write_loss_per_step,
                "wandb_project_name": args.wandb_project_name,
                "wandb_run_name": args.wandb_run_name,
                "wandb_watch_model": args.wandb_watch_model,
                "wandb_model_watch_freq": args.wandb_model_watch_freq,
            },
            "checkpointing": args.checkpointing,
            "checkpointer_config": {
                "checkpoint_path": args.checkpoint_path,
                "checkpoint_freq": args.checkpoint_freq,
                "checkpoint_metric": args.checkpoint_metric,
                "checkpoint_task_metrics": args.checkpoint_task_metrics,
                "checkpoint_runway": args.checkpoint_runway,
                "checkpoint_all": args.checkpoint_all,
                "clear_intermediate_checkpoints": args.clear_intermediate_checkpoints,
                "clear_all_checkpoints": args.clear_all_checkpoints,
            },
        },
    }

    return createBoolDottedDict(config)
