import logging
from collections import Counter

import numpy as np
import torch
from task_config import LABEL_TO_ID

from emmental.data import EmmentalDataset

logger = logging.getLogger(__name__)

CLS = "[CLS]"
SEP = "[SEP]"

SPECIAL_TOKENS = {
    "SUBJ_START": "[unused1]",
    "SUBJ_END": "[unused2]",
    "OBJ_START": "[unused3]",
    "OBJ_END": "[unused4]",
    "SUBJ=ORGANIZATION": "[unused5]",
    "OBJ=PERSON": "[unused6]",
    "SUBJ=PERSON": "[unused7]",
    "OBJ=ORGANIZATION": "[unused8]",
    "OBJ=NUMBER": "[unused9]",
    "OBJ=DATE": "[unused10]",
    "OBJ=NATIONALITY": "[unused11]",
    "OBJ=LOCATION": "[unused12]",
    "OBJ=TITLE": "[unused13]",
    "OBJ=CITY": "[unused14]",
    "OBJ=MISC": "[unused15]",
    "OBJ=COUNTRY": "[unused16]",
    "OBJ=CRIMINAL_CHARGE": "[unused17]",
    "OBJ=RELIGION": "[unused18]",
    "OBJ=DURATION": "[unused19]",
    "OBJ=URL": "[unused20]",
    "OBJ=STATE_OR_PROVINCE": "[unused21]",
    "OBJ=IDEOLOGY": "[unused22]",
    "OBJ=CAUSE_OF_DEATH": "[unused23]",
}


class InputExample(object):
    """A single training/test example for span pair classification."""

    def __init__(
        self,
        guid,
        sentence,
        span1,
        span2,
        ner1,
        ner2,
        ent,
        static_ent,
        type_ent,
        rel_ent,
        label,
    ):
        self.guid = guid
        self.sentence = sentence
        self.span1 = span1
        self.span2 = span2
        self.ner1 = ner1
        self.ner2 = ner2
        self.ent = ent
        self.static_ent = static_ent
        self.type_ent = type_ent
        self.rel_ent = rel_ent
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
        self,
        guid,
        input_ids,
        input_mask,
        segment_ids,
        input_ent_ids,
        input_static_ent_ids,
        input_type_ent_ids,
        input_rel_ent_ids,
        label_id,
    ):
        self.guid = guid
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.input_ent_ids = input_ent_ids
        self.input_static_ent_ids = input_static_ent_ids
        self.input_type_ent_ids = input_type_ent_ids
        self.input_rel_ent_ids = input_rel_ent_ids
        self.label_id = label_id


def convert_token(token):
    """ Convert PTB tokens to normal tokens """
    if token.lower() == "-lrb-":
        return "("
    elif token.lower() == "-rrb-":
        return ")"
    elif token.lower() == "-lsb-":
        return "["
    elif token.lower() == "-rsb-":
        return "]"
    elif token.lower() == "-lcb-":
        return "{"
    elif token.lower() == "-rcb-":
        return "}"
    return token


def create_examples(dataset, encode_first=False):
    """Creates examples."""
    examples = []
    for example in dataset:
        sentence = [convert_token(token) for token in example["token"]]
        assert (
            example["subj_start"] >= 0
            and example["subj_start"] <= example["subj_end"]
            and example["subj_end"] < len(sentence)
        )
        assert (
            example["obj_start"] >= 0
            and example["obj_start"] <= example["obj_end"]
            and example["obj_end"] < len(sentence)
        )
        ent_field = None
        static_ent_field = None
        type_ent_field = None
        rel_ent_field = None
        if "entity_emb_id" in example and encode_first is False:
            ent_field = example["entity_emb_id"]
        if "entity_emb_id_first" in example and encode_first is True:
            ent_field = example["entity_emb_id_first"]
        if "ctx_ent_emb_id" in example and encode_first is False:
            ent_field = example["ctx_ent_emb_id"]
        if "ctx_ent_emb_id_first" in example and encode_first is True:
            ent_field = example["ctx_ent_emb_id_first"]
        if "static_ent_emb_id" in example and encode_first is False:
            static_ent_field = example["static_ent_emb_id"]
        if "static_ent_emb_id_first" in example and encode_first is True:
            static_ent_field = example["static_ent_emb_id_first"]
        if "type_emb_id" in example:
            type_ent_field = example["type_emb_id"]
        if "rel_emb_id" in example:
            rel_ent_field = example["rel_emb_id"]

        examples.append(
            InputExample(
                guid=example["id"],
                sentence=sentence,
                span1=(example["subj_start"], example["subj_end"]),
                span2=(example["obj_start"], example["obj_end"]),
                ner1=example["subj_type"],
                ner2=example["obj_type"],
                ent=[_ + 2 for _ in ent_field] if ent_field is not None else None,
                static_ent=[_ + 2 for _ in static_ent_field]
                if static_ent_field is not None
                else None,
                type_ent=[_ + 2 for _ in type_ent_field]
                if type_ent_field is not None
                else None,
                rel_ent=[_ + 2 for _ in rel_ent_field]
                if rel_ent_field is not None
                else None,
                label=example["relation"],
            )
        )
    return examples


def convert_examples_to_features(
    examples,
    max_seq_length,
    tokenizer,
    special_tokens=None,
    mode="text",
    encode_first=False,
):
    """Loads a data file into a list of `InputBatch`s."""
    if special_tokens is None:
        special_tokens = SPECIAL_TOKENS

    def get_special_token(w):
        if w not in special_tokens:
            special_tokens[w] = "[unused%d]" % (len(special_tokens) + 1)
        return special_tokens[w]

    num_tokens = 0
    num_fit_examples = 0
    num_shown_examples = 0
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens = [CLS]
        ents = [0] if example.ent is not None else None
        static_ents = [0] if example.static_ent is not None else None
        type_ents = [0] if example.type_ent is not None else None
        rel_ents = [0] if example.rel_ent is not None else None

        SUBJECT_START = get_special_token("SUBJ_START")
        SUBJECT_END = get_special_token("SUBJ_END")
        OBJECT_START = get_special_token("OBJ_START")
        OBJECT_END = get_special_token("OBJ_END")
        SUBJECT_NER = get_special_token("SUBJ=%s" % example.ner1)
        OBJECT_NER = get_special_token("OBJ=%s" % example.ner2)

        if mode.startswith("text"):
            for i, token in enumerate(example.sentence):
                if i == example.span1[0]:
                    tokens.append(SUBJECT_START)
                if i == example.span2[0]:
                    tokens.append(OBJECT_START)
                for sub_token in tokenizer.tokenize(token):
                    tokens.append(sub_token)
                if i == example.span1[1]:
                    tokens.append(SUBJECT_END)
                if i == example.span2[1]:
                    tokens.append(OBJECT_END)
            if mode == "text_ner":
                tokens = tokens + [SEP, SUBJECT_NER, SEP, OBJECT_NER, SEP]
            else:
                tokens.append(SEP)
        else:
            subj_tokens = []
            obj_tokens = []
            for i, token in enumerate(example.sentence):
                if i == example.span1[0]:
                    tokens.append(SUBJECT_NER)
                    # Only support ner mode
                    if example.ent is not None:
                        ents.append(example.ent[i])
                    if example.static_ent is not None:
                        static_ents.append(example.static_ent[i])
                    if example.type_ent is not None:
                        type_ents.append(example.type_ent[i])
                    if example.rel_ent is not None:
                        rel_ents.append(example.rel_ent[i])
                if i == example.span2[0]:
                    tokens.append(OBJECT_NER)
                    if example.ent is not None:
                        ents.append(example.ent[i])
                    if example.static_ent is not None:
                        static_ents.append(example.static_ent[i])
                    if example.type_ent is not None:
                        type_ents.append(example.type_ent[i])
                    if example.rel_ent is not None:
                        rel_ents.append(example.rel_ent[i])
                if (i >= example.span1[0]) and (i <= example.span1[1]):
                    for sub_token in tokenizer.tokenize(token):
                        subj_tokens.append(sub_token)
                elif (i >= example.span2[0]) and (i <= example.span2[1]):
                    for sub_token in tokenizer.tokenize(token):
                        obj_tokens.append(sub_token)
                else:
                    for j, sub_token in enumerate(tokenizer.tokenize(token)):
                        tokens.append(sub_token)
                        if example.ent is not None:
                            if encode_first is False:
                                ents.append(example.ent[i])
                            else:
                                if j == 0:
                                    ents.append(example.ent[i])
                                else:
                                    ents.append(1)
                        if example.static_ent is not None:
                            if encode_first is False:
                                static_ents.append(example.static_ent[i])
                            else:
                                if j == 0:
                                    static_ents.append(example.static_ent[i])
                                else:
                                    static_ents.append(1)
                        if example.type_ent is not None:
                            if encode_first is False:
                                type_ents.append(example.type_ent[i])
                            else:
                                if j == 0:
                                    type_ents.append(example.type_ent[i])
                                else:
                                    type_ents.append(1)
                        if example.rel_ent is not None:
                            if encode_first is False:
                                rel_ents.append(example.rel_ent[i])
                            else:
                                if j == 0:
                                    rel_ents.append(example.rel_ent[i])
                                else:
                                    rel_ents.append(1)

            if mode == "ner_text":
                tokens.append(SEP)
                for sub_token in subj_tokens:
                    tokens.append(sub_token)
                tokens.append(SEP)
                for sub_token in obj_tokens:
                    tokens.append(sub_token)
            tokens.append(SEP)
            if example.ent is not None:
                ents.append(0)
            if example.static_ent is not None:
                static_ents.append(0)
            if example.type_ent is not None:
                type_ents.append(0)
            if example.rel_ent is not None:
                rel_ents.append(0)

        num_tokens += len(tokens)

        if len(tokens) > max_seq_length:
            tokens = tokens[:max_seq_length]
            if example.ent is not None:
                ents = ents[:max_seq_length]
            if example.static_ent is not None:
                static_ents = static_ents[:max_seq_length]
            if example.type_ent is not None:
                type_ents = type_ents[:max_seq_length]
            if example.rel_ent is not None:
                rel_ents = rel_ents[:max_seq_length]
        else:
            num_fit_examples += 1

        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding = [0] * (max_seq_length - len(input_ids))
        if ents is not None:
            input_ent_ids = ents
        if static_ents is not None:
            input_static_ent_ids = static_ents
        if type_ents is not None:
            input_type_ent_ids = type_ents
        if rel_ents is not None:
            input_rel_ent_ids = rel_ents

        input_ids += padding
        input_mask += padding
        segment_ids += padding
        if ents is not None:
            input_ent_ids += padding
        else:
            input_ent_ids = None
        if static_ents is not None:
            input_static_ent_ids += padding
        else:
            input_static_ent_ids = None
        if type_ents is not None:
            input_type_ent_ids += padding
        else:
            input_type_ent_ids = None
        if rel_ents is not None:
            input_rel_ent_ids += padding
        else:
            input_rel_ent_ids = None

        label_id = LABEL_TO_ID[example.label]
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        if ents is not None:
            assert len(input_ent_ids) == max_seq_length
        if static_ents is not None:
            assert len(input_static_ent_ids) == max_seq_length
        if type_ents is not None:
            assert len(input_type_ent_ids) == max_seq_length
        if rel_ents is not None:
            assert len(input_rel_ent_ids) == max_seq_length

        if num_shown_examples < 20:
            if (ex_index < 5) or (label_id > 0):
                num_shown_examples += 1
                logger.info("*** Example ***")
                logger.info("guid:        %s" % (example.guid))
                logger.info(
                    "examples:    %s" % " ".join([str(x) for x in example.sentence])
                )
                logger.info("tokens:      %s" % " ".join([str(x) for x in tokens]))
                if example.ent is not None:
                    logger.info(
                        "ents:        %s" % " ".join([str(x) for x in example.ent])
                    )
                logger.info("input_ids:   %s" % " ".join([str(x) for x in input_ids]))
                logger.info("input_mask:  %s" % " ".join([str(x) for x in input_mask]))
                logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                if ents is not None:
                    logger.info(
                        "input_ent_ids: %s" % " ".join([str(x) for x in input_ent_ids])
                    )
                if static_ents is not None:
                    logger.info(
                        "input_static_ent_ids: %s"
                        % " ".join([str(x) for x in input_static_ent_ids])
                    )
                if type_ents is not None:
                    logger.info(
                        "input_type_ent_ids: %s"
                        % " ".join([str(x) for x in input_type_ent_ids])
                    )
                if rel_ents is not None:
                    logger.info(
                        "input_rel_ent_ids: %s"
                        % " ".join([str(x) for x in input_rel_ent_ids])
                    )

                logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(
                guid=example.guid,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                input_ent_ids=input_ent_ids,
                input_static_ent_ids=input_static_ent_ids,
                input_type_ent_ids=input_type_ent_ids,
                input_rel_ent_ids=input_rel_ent_ids,
                label_id=label_id,
            )
        )
    logger.info("Average #tokens: %.2f" % (num_tokens * 1.0 / len(examples)))
    logger.info(
        "%d (%.2f %%) examples can fit max_seq_length = %d"
        % (num_fit_examples, num_fit_examples * 100.0 / len(examples), max_seq_length)
    )
    logger.info(f"{special_tokens}")
    return features


def get_labels(examples):
    count = Counter()
    for example in examples:
        count[example.label] += 1
    logger.info("%d labels" % len(count))
    # Make sure the negative label is alwyas 0
    for label, count in count.most_common():
        logger.info("%s: %.2f%%" % (label, count * 100.0 / len(examples)))


class TACREDDataset(EmmentalDataset):
    """Dataset to load TACRED dataset."""

    def __init__(
        self,
        name,
        dataset,
        tokenizer,
        split="train",
        mode="text",
        max_seq_length=128,
        bert_mode="base",
        encode_first=False,
    ):
        X_dict, Y_dict = (
            {
                "guids": [],
                "token_ids": [],
                "token_masks": [],
                "token_ent_ids": [],
                "token_static_ent_ids": [],
                "token_type_ent_ids": [],
                "token_rel_ent_ids": [],
                "token_segments": [],
            },
            {"labels": []},
        )
        examples = create_examples(dataset, encode_first)
        logger.info(f"{split} set stats:")
        get_labels(examples)
        features = convert_examples_to_features(
            examples,
            max_seq_length=max_seq_length,
            tokenizer=tokenizer,
            special_tokens=SPECIAL_TOKENS,
            mode=mode,
            encode_first=encode_first,
        )

        for i, feature in enumerate(features):
            X_dict["guids"].append(feature.guid)
            X_dict["token_ids"].append(torch.LongTensor(feature.input_ids))
            X_dict["token_masks"].append(torch.LongTensor(feature.input_mask))
            X_dict["token_segments"].append(torch.LongTensor(feature.segment_ids))
            if feature.input_ent_ids is not None:
                X_dict["token_ent_ids"].append(torch.LongTensor(feature.input_ent_ids))
            else:
                X_dict["token_ent_ids"].append(None)
            if feature.input_static_ent_ids is not None:
                X_dict["token_static_ent_ids"].append(
                    torch.LongTensor(feature.input_static_ent_ids)
                )
            else:
                X_dict["token_static_ent_ids"].append(None)
            if feature.input_type_ent_ids is not None:
                X_dict["token_type_ent_ids"].append(
                    torch.LongTensor(feature.input_type_ent_ids)
                )
            else:
                X_dict["token_type_ent_ids"].append(None)
            if feature.input_rel_ent_ids is not None:
                X_dict["token_rel_ent_ids"].append(
                    torch.LongTensor(feature.input_rel_ent_ids)
                )
            else:
                X_dict["token_rel_ent_ids"].append(None)

            Y_dict["labels"].append(feature.label_id)

        Y_dict["labels"] = torch.from_numpy(np.array(Y_dict["labels"]))

        super().__init__(name, X_dict=X_dict, Y_dict=Y_dict, uid="guids")

    def __getitem__(self, index):
        r"""Get item by index.

        Args:
          index(index): The index of the item.

        Returns:
          Tuple[Dict[str, Any], Dict[str, Tensor]]: Tuple of x_dict and y_dict

        """
        x_dict = {name: feature[index] for name, feature in self.X_dict.items()}
        y_dict = {name: label[index] for name, label in self.Y_dict.items()}
        return x_dict, y_dict
