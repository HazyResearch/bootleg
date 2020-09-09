import os
import json
import ujson

from bootleg.utils import train_utils

class StatusReporter:
    def __init__(self, args, logger, is_writer, max_epochs, total_steps_per_epoch, is_eval):
        self.format_str = 'epoch {}, step {}/{}, loss = {:.6f} ({:.3f} sec/batch | {:.3f} sec/batch LOAD), lr: {:.6f}'
        self.logger = logger
        self.is_writer = is_writer
        self.max_epochs = max_epochs
        self.total_steps_per_epoch = total_steps_per_epoch
        self.dev_files = {}
        self.loss_file = ""
        self.test_files = {}
        if not is_eval:
            self.dev_files = self.setup_dev_files(args)
            self.loss_file = self.setup_loss_file(args)
        else:
            self.test_files = self.setup_test_files(args)

    def setup_test_files(self, args):
        test_files = {}
        save_folder = train_utils.get_save_folder(args.run_config)
        test_file_tag = args.data_config.test_dataset.file.split('.jsonl')[0]
        test_file = test_file_tag + "_test_results"
        test_file += train_utils.get_file_suffix(args)
        test_file += '.jsonl'
        test_file = os.path.join(save_folder, test_file)
        # Clear old file
        open(test_file, 'w').close()
        test_files[args.data_config.test_dataset.file] = test_file
        return test_files

    def setup_dev_files(self, args):
        dev_files = {}
        save_folder = train_utils.get_save_folder(args.run_config)
        dev_file_tag = args.data_config.dev_dataset.file.split('.jsonl')[0]
        dev_file = dev_file_tag + "_dev_results"
        dev_file += train_utils.get_file_suffix(args)
        dev_file += '.jsonl'
        dev_file = os.path.join(save_folder, dev_file)
        # Clear old file
        open(dev_file, 'w').close()
        dev_files[args.data_config.dev_dataset.file] = dev_file
        return dev_files

    def setup_loss_file(self, args):
        save_folder = train_utils.get_save_folder(args.run_config)
        loss_file = "loss_results"
        loss_file += train_utils.get_file_suffix(args)
        loss_file += '.jsonl'
        loss_file = os.path.join(save_folder, loss_file)
        open(loss_file, 'w').close()
        return loss_file

    def step_status(self, epoch, step, loss_pack, time, load_time, lr):
        self.logger.info(self.format_str.format(epoch,
                                                step,
                                                self.total_steps_per_epoch*self.max_epochs,
                                                loss_pack.loss.data.item(),
                                                time,
                                                load_time,
                                                lr))
        if self.is_writer:
            self.dump_loss(epoch, step, loss_pack.loss_dict)
        return

    def dump_loss(self, epoch, step, loss_dict):
        if self.is_writer:
            with open(self.loss_file, 'a') as f:
                loss_dict["epoch"] = epoch
                loss_dict["step"] = step
                ujson.dump(loss_dict, f)
                f.write('\n')
        return

    def dump_results(self, eval_results, pretty_results_printing, file, is_test):
        if self.is_writer:
            if is_test:
                assert file in self.test_files, f'The dump file {file} is not in our possible files from {self.test_files.keys()}'
            else:
                assert file in self.dev_files, f'The dump file {file} is not in our possible files from {self.dev_files.keys()}'
            if not is_test:
                file = self.dev_files[file]
                file_mode = 'a'
            else:
                file = self.test_files[file]
                file_mode = 'a'
            with open(file, file_mode) as f:
                # json (rather than ujson) allows nans which may occur if an eval slice is empty
                json.dump(eval_results, f)
                f.write('\n')
        return