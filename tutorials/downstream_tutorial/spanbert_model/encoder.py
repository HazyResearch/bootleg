from torch import nn


class Encoder(nn.Module):
    def __init__(self, encoder, pooler, freeze=False):
        super().__init__()

        self.encoder = encoder
        self.pooler = pooler

        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.pooler.parameters():
                param.requires_grad = False

    def forward(self, embedding, attention_mask, output_all_encoded_layers=True):
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        encoded_layers = self.encoder(
            embedding,
            extended_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
        )
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]

        return encoded_layers, pooled_output
