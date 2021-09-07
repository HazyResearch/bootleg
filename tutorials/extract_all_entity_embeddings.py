import tempfile
from pathlib import Path

import argh
import numpy as np
from rich.console import Console
from rich.progress import Progress
from transformers.tokenization_utils import _is_control

from bootleg.end2end.bootleg_annotator import BootlegAnnotator
from bootleg.symbols.entity_profile import EntityProfile
from bootleg.utils.utils import get_lnrm

console = Console(soft_wrap=True)
bert_dir = tempfile.TemporaryDirectory().name


def get_mention(title):
    return get_lnrm(title, strip=True, lower=True)


@argh.arg("--bootleg_data", help="Path for saving new data")
@argh.arg("--model_name", help="Model name")
@argh.arg("--save_emb_path", help="Path to save embs")
@argh.arg("--device", help="Device for Bootleg")
@argh.arg("--batch_size", help="Batch size")
def main(
    bootleg_data="tutorial_data",
    model_name="bootleg_uncased",
    save_emb_path="tutorial_data/embeddings",
    device=0,
    batch_size=65536,
):
    bootleg_data = Path(bootleg_data)
    save_emb_path = Path(save_emb_path)
    save_emb_path.mkdir(parents=True, exist_ok=True)

    # Load annotator
    ann = BootlegAnnotator(
        cache_dir=bootleg_data,
        model_name=model_name,
        device=device,
        return_embs=True,
        verbose=False,
    )
    config = ann.config
    console.print(f"Loaded annotator")

    # Loading ep
    ep = EntityProfile.load_from_cache(Path(bootleg_data) / "data/entity_db")
    qid2aliases = {}
    for al in ep.get_all_mentions():
        for qid in ep.get_qid_cands(al):
            if qid not in qid2aliases:
                qid2aliases[qid] = set()
            qid2aliases[qid].add(al)

    console.print(f"Loaded entity profile")

    total_shape = config["model_config"]["hidden_size"]

    all_qids = sorted(list(ep.get_all_qids()), reverse=True)
    # +1 for the unk entity
    all_ent_embeddings = np.zeros((len(all_qids) + 1, total_shape))
    i = 0
    failed_qids = []
    with Progress() as progress:
        task = progress.add_task("Iterating qids", total=len(all_qids))
        # Iterate over all qids, sending through forward pass of the embedding layers that are not contextual.
        # Collect into embedding matrix and save.
        while i < len(all_qids):
            print(
                f"Staring {i} with bs {batch_size} with {len(all_qids)} total for itrs {len(all_qids)/batch_size}"
            )
            batch_qids = all_qids[i : i + batch_size]
            to_skip = False
            if i < -1:
                to_skip = True
            i += batch_size
            if not to_skip:
                # For each QID, we want to customize the candidates/mentions so there is one candidate
                # and the sentence is the title
                # This should give the "purest" form of a contextualized entity embeddings
                extracted_exs = []
                for q in batch_qids:
                    title = ep.get_title(q)

                    # Set mention to be the title
                    men_title = get_mention(title)

                    # Ensure title isn't weird character that has no mention -> breaks sentence parsing
                    if len(men_title) <= 0 or all(_is_control(c) for c in men_title):
                        # Try other mentions
                        for al in qid2aliases.get(q, []):
                            men_title = get_mention(al)
                            if len(men_title) > 0 and not all(
                                _is_control(c) for c in men_title
                            ):
                                title = al
                                men_title = get_mention(title)
                                console.print(
                                    f"Found new title for {ep.get_title(q)} of {title} with {men_title} men"
                                )
                                break
                        # If still no good title, set it to be the description
                        if len(men_title) <= 0 or all(
                            _is_control(c) for c in men_title
                        ):
                            console.print(f"Not found new title for {ep.get_title(q)}.")
                            console.print(
                                f"Not found new title or desc for {ep.get_title(q)}. Setting to unk."
                            )
                            title = "unknown"
                            men_title = get_mention(title)
                    item = {
                        "sentence": title,
                        "aliases": [men_title],
                        "spans": [[0, len(title.split())]],
                        "cands": [[q]],
                    }
                    extracted_exs.append(item)
                t = 0
                for ex in extracted_exs:
                    if len(ex["sentence"]) <= 2:
                        print("Maybe Bad", ex)
                    if t < 5:
                        print(ex)
                        t += 1
                batch_eids = [ep.get_eid(q) for q in batch_qids]
                out_dict = ann.label_mentions(extracted_examples=extracted_exs)
                pred_qids = out_dict["qids"]
                pred_embs = out_dict["embs"]
                for qid, pred_qids, eid, embs in zip(
                    batch_qids, pred_qids, batch_eids, pred_embs
                ):
                    assert len(pred_qids) == len(embs)
                    if len(pred_qids) != 1 or pred_qids[0] != qid:
                        failed_qids.append(qid)
                        console.print(
                            "Failed",
                            qid,
                            pred_qids,
                            ep.get_title(qid),
                            [ep.get_title(q) for q in pred_qids],
                        )
                    else:
                        all_ent_embeddings[eid] = embs[0]
                del out_dict
            progress.update(task, advance=len(batch_qids))

    save_path = save_emb_path / f"ent_embeddings_{model_name}.npy"
    np.save(str(save_path), all_ent_embeddings)
    console.print(
        "Embs Shape",
        all_ent_embeddings.shape,
        "Num Qids",
        len(all_qids),
        "Saved To",
        save_path,
    )


if __name__ == "__main__":
    argh.dispatch_command(main)
