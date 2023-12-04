
import ast
import torch
import torch.nn.functional as F
import torch.nn as nn

class VectorQuantization(torch.nn.Module):

    """

    Implementation of Vector Quantization using the Gumbel-Softmax.

    We follow the descriptions of:
    [1] van den Oord et al. - Neural Discrete Representation Learning
    [2] Schneider et al.    - Wav2Vec: Unsupervised Pre-Training for Speech Recognition

    - Adapted from https://github.com/pytorch/fairseq/blob/main/fairseq/modules/gumbel_vector_quantizer.py

    Args:
        input_dim: input dimension (channels)
        emb_dim: dimensionality of the resulting quantized vector
        groups_dim: number of groups for vector quantization
        entries_dim: number of quantized vectors per group
        tau: temperature for training. this should be a tuple of 3 elements: (start, stop, decay factor)
        time_first: if true, expect input in BxTxC format, otherwise in BxCxT

    """

    def __init__(self, parameters):
        super(VectorQuantization, self).__init__()

        assert (
                parameters["emb_dim"] % parameters["groups_dim"] == 0
        ), f"dim {parameters['emb_dim']} must be divisible by groups {parameters['groups_dim']} for concatenation"

        self.input_dim = parameters["input_dim"]
        self.groups_dim = parameters["groups_dim"]
        self.entries_dim = parameters["entries_dim"]

        self.time_first = parameters["time_first"]
        self.return_q   = parameters["return_q"]

        self.emb_dim = parameters["emb_dim"]
        self.true_emb_dim = parameters["emb_dim"] // parameters["groups_dim"]

        self.codebook = nn.Parameter(torch.FloatTensor(1, parameters["groups_dim"] * parameters["entries_dim"],
                                                       self.true_emb_dim), requires_grad=True)
        nn.init.uniform_(self.codebook)

        self.weight_proj = nn.Linear(self.input_dim, parameters["groups_dim"] * parameters["entries_dim"])
        nn.init.normal_(self.weight_proj.weight, mean=0, std=1)
        nn.init.zeros_(self.weight_proj.bias)

        if isinstance(parameters["tau"], str):
            tau = ast.literal_eval(parameters["tau"])
        else:
            tau = parameters["tau"]

        assert len(tau) == 3, f"{tau}, {len(tau)}"

        self.max_tau, self.min_tau, self.tau_decay = tau
        self.curr_tau = self.max_tau
        self.codebook_indices = None

    def set_num_updates(self, num_updates):
        self.curr_tau = max(
            self.max_tau * self.tau_decay**num_updates, self.min_tau
        )

    def forward(self, X, num_updates=None):

        # If the number of iterations is passed in to the function,
        # update the value of tau
        if num_updates is not None and self.training:
            self.set_num_updates(num_updates)

        # Get dims
        n_batch, n_timesteps, n_feats = X.shape

        # Project input into codebook dimension
        X = self.weight_proj(X)  # B*T x F -> B*T x G*V

        # B*T x G*V -> B*T*G x V 
        X = X.view(n_batch * self.groups_dim, -1)

        # Do "hard" argmax
        _, k = X.max(-1)
        hard_x = X.new_zeros(*X.shape).scatter_(-1, k.view(-1, 1), 1.0)

        # B*T x G*V -> B*T x G x V 
        hard_x = hard_x.view(n_batch * n_timesteps, self.groups_dim, -1)

        # Save perplexity and other variables to later compute the diversity loss during training
        result = {"num_vars": self.entries_dim * self.groups_dim, "tau": self.curr_tau}
        if self.training:
            hard_probs = torch.mean(hard_x.float(), dim=0)
            avg_probs  = torch.softmax(X.view(n_batch * n_timesteps, self.groups_dim, -1).float(), dim=-1).mean(dim=0)

            result["code_perplexity"] = torch.exp(-torch.sum(hard_probs * torch.log(hard_probs + 1e-7), dim=-1)).sum()
            result["prob_perplexity"] = torch.exp(-torch.sum(avg_probs  * torch.log(avg_probs  + 1e-7), dim=-1)).sum()

        # If in training - Use Gumbel-Softmax
        if self.training:
            X = F.gumbel_softmax(X.float(), tau=self.curr_tau, hard=True).type_as(X)

            if self.return_q:
                # Save differentiable quantised vectors
                result["Q"] = X.clone()
        else:
            # Else: use argmax
            X = hard_x
            
            if self.return_q:
                result["Q"] = X.clone()

        # B*T x G x V -> B*T x G*V
        X = X.view(n_batch * n_timesteps, -1)

        # Apply indices to codebook selection B*T x G*V x d/G
        X = X.unsqueeze(-1) * self.codebook

        # B*T x V*G x d/G -> B*T x G x V x d/G
        X = X.view(n_batch * n_timesteps, self.groups_dim, self.entries_dim, -1)

        # B*T x G x V x d/G -> B*T x G x d/G
        X = X.sum(-2)

        # B*T x G x d -> B x T x G*d/G
        X = X.view(n_batch, n_timesteps, -1)

        if not self.time_first:
            X = X.transpose(1, 2)  # BTC -> BCT

        return X, result

