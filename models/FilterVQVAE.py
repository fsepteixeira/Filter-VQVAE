
import torch

import torch.nn.functional as F

from .       import GatedFeedForward, FeedForward
from modules import VectorQuantization, EmbeddingProjection, CodeProjection

class FilterVQVAE(torch.nn.Module):

    def __init__(self, params):

        super(FilterVQVAE, self).__init__()

        if params["gated"]:
            self.encoder = GatedFeedForward(params["encoder"])
            self.decoder = GatedFeedForward(params["decoder"])
        else:
            self.encoder = FeedForward(params["encoder"])
            self.decoder = FeedForward(params["decoder"])

        self.output_layer = torch.nn.Linear(params["output_layer"]["in_dim"],
                                            params["output_layer"]["out_dim"])

        self.vector_quantizer = VectorQuantization(params["quantizer"])
        self.embedding_proj   = EmbeddingProjection(params["emb_projection"])
        self.code_proj        = CodeProjection(params["q_projection"])

        self.params = params

    def forward(self, X, embedding, dropout=None, num_updates=None):

        # Encoder
        e = self.encoder(X, dropout)

        # Vector Quantizer
        q, r = self.vector_quantizer(e.unsqueeze(dim=1), num_updates=num_updates)
        q = q.squeeze(dim=1)

        # Project soft label to a more meaningful space
        embedding = self.embedding_proj(embedding)
        q         = self.code_proj(q)

        # Concatenate with output of quantizer
        c = torch.cat([q, embedding], dim=-1)

        # Decoder conditioned on soft labels y
        d = self.decoder(c, dropout)

        # Output projection layer
        Y = self.output_layer(d)

        return Y, e, q, r

