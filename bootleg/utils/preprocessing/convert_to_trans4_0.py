import glob
import shutil
import torch
import os
import ujson
import argparse

MAX_LEN = 512
BERT_DIM = 768

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--bert_model', type=str, default='bert-base-cased')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print(ujson.dumps(args, indent=4))

    # Find all models
    all_models = glob.glob(os.path.join(args.model_dir, "*.pt"))
    saved_models = os.path.join(args.model_dir, "saved_trans_3.0.1")
    if not os.path.exists(saved_models):
        os.makedirs(saved_models)

    for mod in all_models:
        name = os.path.basename(mod)
        new_file = os.path.join(saved_models, name)
        print(f"Moving {mod} to {new_file}")
        shutil.copyfile(mod, new_file)

    # Load and convert
    for mod in all_models:
        model_dict = torch.load(mod, lambda storage, loc: storage)
        mod_state_dict = model_dict["model"]
        if "emb_layer.word_emb.embeddings.word_embeddings.weight" in mod_state_dict:
            print("SWAP WORD EMB")
            new_position_ids = torch.arange(MAX_LEN).expand((1, -1))
            mod_state_dict["emb_layer.word_emb.embeddings.position_ids"] = new_position_ids
        elif "module.emb_layer.word_emb.embeddings.word_embeddings.weight" in mod_state_dict:
            print("SWAP WORD EMB")
            new_position_ids = torch.arange(MAX_LEN).expand((1, -1))
            mod_state_dict["module.emb_layer.word_emb.embeddings.position_ids"] = new_position_ids
        if "emb_layer.entity_embs.avg_title_proj.word_emb.embeddings.word_embeddings.weight" in mod_state_dict:
            print("SWAP TITLE WORD EMB")
            new_position_ids = torch.arange(MAX_LEN).expand((1, -1))
            mod_state_dict["emb_layer.entity_embs.avg_title_proj.word_emb.embeddings.position_ids"] = new_position_ids
        elif "module.emb_layer.entity_embs.avg_title_proj.word_emb.embeddings.word_embeddings.weight" in mod_state_dict:
            print("SWAP TITLE WORD EMB")
            new_position_ids = torch.arange(MAX_LEN).expand((1, -1))
            mod_state_dict["module.emb_layer.entity_embs.avg_title_proj.word_emb.embeddings.position_ids"] = new_position_ids
        model_dict["model"] = mod_state_dict
        print(f"Saving new model in {mod}")
        torch.save(model_dict, mod)

if __name__ == '__main__':
    main()