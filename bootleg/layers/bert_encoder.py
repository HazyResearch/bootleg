import os

import torch
from torch import nn
from transformers import BertModel


class BertEncoder(nn.Module):
    """Bert sentence encoder class.

    This also handles any entity embedding that requires a forward pass through the BERT encoder.

    Args:
        emb_args: embedding args
        output_size: output size for sentence projection (default None)
    """

    def __init__(self, emb_args, output_size=None):
        super().__init__()
        cache_dir = emb_args.cache_dir
        bert_model_name = emb_args.bert_model
        self.num_layers = emb_args.layers
        self.use_sent_proj = emb_args.use_sent_proj
        self.pad_id = 0
        self.dim = 768
        self.output_size = (
            output_size
            if (output_size is not None and emb_args.use_sent_proj)
            else self.dim
        )
        # Create cache directory if not exists
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        self.bert_model = BertModel.from_pretrained(
            bert_model_name, cache_dir=cache_dir
        )
        self.bert_model.encoder.layer = self.bert_model.encoder.layer[: self.num_layers]
        self.requires_grad = not emb_args.freeze
        if not self.requires_grad:
            self.bert_model.eval()
            for name, param in self.bert_model.named_parameters():
                param.requires_grad = False

        if self.use_sent_proj:
            self.sent_proj = nn.Linear(self.dim, output_size)

        # Embedding layers that output token ids for BERT. Due to how DDP handles calls to one module multiple times
        # with how task flows are done in Emmental, we encapsulate all BERT calls inside this forward() call.
        # Therefore, this class must handle the embedding forward() and postprocessing()
        self.emb_objs = torch.nn.ModuleList()

    def add_embedding(self, emb_obj):
        """Add an entity embedding class that requires BERT forward calls.

        Args:
            emb_obj: embedding class

        Returns:
        """
        assert hasattr(emb_obj, "postprocess_embedding"), (
            f"To add an embedding layer to BERT, it requies you to define a forward() and "
            f"postprocess_embedding() method. postprocess_embedding() is not defined."
        )
        assert hasattr(emb_obj, "forward"), (
            f"To add an embedding layer to BERT, it requies you to define a forward() and "
            f"postprocess_embedding() method. forward() is not defined."
        )
        self.emb_objs.append(emb_obj)

    def bert_forward(self, token_ids, token_type_ids, attention_mask, requires_grad):
        """Forward pass of the sentence embedding.

        Args:
            token_ids: word token ids (B x N)
            token_type_ids: token type ids (B x N)
            attention_mask: attention mask (B x N)
            requires_grad: requires gradients or not

        Returns: output sentence embedding (B x N x L)
        """
        if requires_grad:
            output = self.bert_model(
                token_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
            )[0]
        else:
            with torch.no_grad():
                output = self.bert_model(
                    token_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                )[0]
        return output

    def forward(self, entity_cand_eid, token_ids):
        """Model forward pass.

        Args:
            entity_cand_eid: entity candidate EIDs (B x M x K)
            token_ids: word token ids (B x N)

        Returns: sentence embedding (B x N x L), downstream sentence embedding mask (B x N),
                *all entity embeddings fed through BERT (B x M x K x dim)
        """
        # Handle the sentence embedding
        token_type_ids = torch.zeros_like(token_ids)
        attention_mask = (token_ids != self.pad_id).long()
        requires_grad = self.requires_grad and self.training
        downstream_output = self.bert_forward(
            token_ids, token_type_ids, attention_mask, requires_grad
        )
        if self.use_sent_proj:
            downstream_output = self.sent_proj(downstream_output)
        assert downstream_output.shape[-1] == self.output_size
        # This mask is for downstream pytorch multiheadattention
        # This assumes that TRUE means MASK (aka IGNORE). For the sentence embedding,
        # the mask therefore is if an index is equal to the pad id
        # Note: This mask cannot be used for a BERT model as they use the reverse mask.
        downstream_mask = token_ids == self.pad_id

        # Handle the entity embedding layers
        embedding_layer_outputs = []
        for emb_obj in self.emb_objs:
            # Call forward
            for_bert_outputs = emb_obj(entity_cand_eid, batch_on_the_fly_data={})

            if type(for_bert_outputs) is torch.Tensor:
                for_bert_outputs = (for_bert_outputs,)

            token_ids = for_bert_outputs[0]
            if len(for_bert_outputs) > 1:
                token_type_ids = for_bert_outputs[1]
                assert type(token_type_ids) is torch.Tensor
            else:
                token_type_ids = torch.zeros_like(token_ids)
            if len(for_bert_outputs) > 2:
                attention_mask = for_bert_outputs[2]
                assert type(attention_mask) is torch.Tensor
            else:
                attention_mask = (token_ids != self.pad_id).long()
            if len(for_bert_outputs) > 3:
                assert type(for_bert_outputs[3]) is bool
                requires_grad = (
                    self.requires_grad and self.training and for_bert_outputs[3]
                )

            output = self.bert_forward(
                token_ids, token_type_ids, attention_mask, requires_grad
            )
            output_mask = token_ids == self.pad_id
            # Call postprocess
            final_embedding = emb_obj.postprocess_embedding(
                output, output_mask, *for_bert_outputs[4:]
            )
            embedding_layer_outputs.append(final_embedding)

        return (downstream_output, downstream_mask, *embedding_layer_outputs)
