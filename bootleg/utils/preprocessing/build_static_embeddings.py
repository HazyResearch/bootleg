"""Builds our static title embedding for each entity.

The output of this is torch saved pt file to be read in by our StaticEmb class.

```
ent_embeddings:
   - key: title_static
     load_class: StaticEmb
     freeze: false # Freeze the projection layer or not
     cpu: true
     args:
       emb_file: <path to saved pt file>
       proj: 256
```
"""

import argparse
import os

import torch
import ujson
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

from bootleg.symbols.entity_symbols import EntitySymbols

MAX_LEN = 512
BERT_DIM = 768


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method", type=str, choices=["avg_title"], default="avg_title"
    )
    parser.add_argument(
        "--entity_dir",
        type=str,
        default="/dfs/scratch0/lorr1/projects/bootleg/data/wiki_title_0122/entity_db",
        help="Path to entities inside data_dir",
    )
    parser.add_argument(
        "--entity_map_dir",
        type=str,
        default="entity_mappings",
        help="Path to entities inside data_dir",
    )
    parser.add_argument(
        "--alias_cand_map",
        type=str,
        default="alias2qids.json",
        help="Path to alias candidate map",
    )
    parser.add_argument(
        "--bert_model", type=str, default="bert-base-cased", help="Bert model"
    )
    parser.add_argument(
        "--word_model_cache",
        type=str,
        default="/dfs/scratch0/lorr1/projects/bootleg-data/embs/pretrained_bert_models",
        help="Path to saved model",
    )
    parser.add_argument(
        "--save_file", type=str, required=True, help="Path to save embedding file"
    )
    parser.add_argument("--batch_size", type=int, default=2056)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output_method", default="pt", choices=["pt", "json"])

    args = parser.parse_args()
    return args


def average_titles(input_ids, embeddings):
    num_valid = (input_ids != 0).sum(-1)
    return embeddings.sum(1) / num_valid.unsqueeze(1)


def build_title_table(cpu, batch_size, model, tokenizer, entity_symbols):
    """Builds the table of the word indices associated with each title."""
    entity2avgtitle = torch.zeros(
        entity_symbols.num_entities_with_pad_and_nocand, BERT_DIM
    )
    titles = []
    eids = []
    for q in tqdm(
        entity_symbols.get_all_qids(),
        total=len(entity_symbols.get_all_titles()),
        desc="Itearting over entities",
    ):
        eids.append(entity_symbols.get_eid(q))
        titles.append(entity_symbols.get_title(q))

    assert len(eids) == len(titles)
    for i in tqdm(range(0, len(titles), batch_size)):
        batch_eids = eids[i : i + batch_size]
        batch_titles = titles[i : i + batch_size]
        batch_inputs = tokenizer(
            batch_titles, padding=True, truncation=True, return_tensors="pt"
        )
        inputs = batch_inputs["input_ids"]
        attention_mask = batch_inputs["attention_mask"]
        inputs = inputs.to(model.device)
        attention_mask = attention_mask.to(model.device)
        # model() returns tuple of (last layer of embeddings, pooled output)
        with torch.no_grad():
            outputs = model(inputs, attention_mask=attention_mask)[0]
        assert list(outputs.shape) == [len(batch_titles), inputs.shape[1], BERT_DIM]
        outputs[inputs == 0] = 0
        assert all(outputs[(1 - attention_mask).bool()].sum(-1) == 0)
        entity2avgtitle[batch_eids] = average_titles(inputs, outputs).to("cpu")
    return entity2avgtitle


def main():
    args = parse_args()
    print(ujson.dumps(vars(args), indent=4))
    entity_symbols = EntitySymbols.load_from_cache(
        os.path.join(args.entity_dir, args.entity_map_dir),
        alias_cand_map_file=args.alias_cand_map,
    )
    print("DO LOWERCASE IS", "uncased" in args.bert_model)
    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model,
        do_lower_case="uncased" in args.bert_model,
        cache_dir=args.word_model_cache,
    )
    model = BertModel.from_pretrained(
        args.bert_model,
        cache_dir=args.word_model_cache,
        output_attentions=False,
        output_hidden_states=False,
    )
    if not args.cpu:
        model = model.to("cuda")
    model.eval()

    entity2avgtitle = build_title_table(
        args.cpu, args.batch_size, model, tokenizer, entity_symbols
    )
    save_fold = os.path.dirname(args.save_file)
    if len(save_fold) > 0:
        if not os.path.exists(save_fold):
            os.makedirs(save_fold)
    if args.output_method == "pt":
        save_obj = (entity_symbols.get_qid2eid(), entity2avgtitle)
        torch.save(obj=save_obj, f=args.save_file)
    else:
        res = {}
        for qid in tqdm(entity_symbols.get_all_qids(), desc="Building final json"):
            eid = entity_symbols.get_eid(qid)
            res[qid] = entity2avgtitle[eid].tolist()
        with open(args.save_file, "w") as out_f:
            ujson.dump(res, out_f)
    print(f"Done!")


if __name__ == "__main__":
    main()
