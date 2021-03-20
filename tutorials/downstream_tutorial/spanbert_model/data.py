import logging
import os

from dataset import TACREDDataset
from pytorch_pretrained_bert import BertTokenizer
from utils import load_json

from emmental.data import EmmentalDataLoader

logger = logging.getLogger(__name__)


def get_dataloaders(args):
    task = "TACRED"

    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=True if "uncased" in args.bert_model else False
    )

    datasets = {}
    for split in ["train", "dev", "test"]:
        if split == "train":
            logger.info(
                f"Loading {split} from "
                f"{os.path.join(args.data_dir, f'{split}.json')}."
            )
            data = load_json(os.path.join(args.data_dir, f"{split}.json"))
        else:
            logger.info(
                f"Loading {split} from "
                f"{os.path.join(args.data_dir, f'{split}.json')}."
            )
            data = load_json(os.path.join(args.data_dir, f"{split}.json"))

        datasets[split] = TACREDDataset(
            task,
            data,
            tokenizer=tokenizer,
            split=split,
            mode=args.feature_mode,
            max_seq_length=args.max_seq_length,
            encode_first=args.encode_first,
        )

    dataloaders = []
    for split, dataset in datasets.items():
        dataloaders.append(
            EmmentalDataLoader(
                task_to_label_dict={task: "labels"},
                dataset=dataset,
                split=split,
                shuffle=True if split in ["train"] else False,
                batch_size=args.batch_size
                if split in args.train_split or args.valid_batch_size is None
                else args.valid_batch_size,
                num_workers=4,
            )
        )
        logger.info(
            f"Built dataloader for {split} set with {len(dataset)} "
            f"samples (Shuffle={split in args.train_split}, "
            f"Batch size={dataloaders[-1].batch_size})."
        )

    return dataloaders
