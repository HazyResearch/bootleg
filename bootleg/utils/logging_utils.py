import logging
import multiprocessing
import os

from bootleg.utils import train_utils


def get_log_name(args, mode):
    log_name = os.path.join(train_utils.get_save_folder(args.run_config), f"log_{mode}")
    print(train_utils.get_file_suffix(args))
    log_name += train_utils.get_file_suffix(args)
    log_name += f'_gpu{args.run_config.gpu}'
    return log_name


def create_logger(args, mode):
    if args.run_config.distributed:
        logger = multiprocessing.get_logger()
    else:
        logger = logging.getLogger("bootleg")

    # set logging level
    numeric_level = getattr(logging, args.run_config.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.run_config.loglevel.upper())
    logger.setLevel(numeric_level)
    # do not propagate messages to the root logger
    logger.propagate = False
    log_name = get_log_name(args, mode)
    if not os.path.exists(log_name): os.system("touch " + log_name)
    if not logger.hasHandlers():
        formatter = logging.Formatter('%(asctime)s %(message)s')
        fh = logging.FileHandler(log_name, mode='w' if mode == 'train' else 'a')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        # only print the stream for the first GPU
        if args.run_config.gpu == 0:
            sh = logging.StreamHandler()
            sh.setFormatter(formatter)
            logger.addHandler(sh)
    else:
        print('here')
        exit()
    return logger

def get_logger(args):
    if args.run_config.distributed:
        return multiprocessing.get_logger()
    else:
        return logging.getLogger("bootleg")