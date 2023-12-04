
import torch
from modules import LinearBlock, \
                    GradientReversalLayer

class EmbeddingClassifier(torch.nn.Module):

    def __init__(self, parameters, load_checkpoint=False):

        super(EmbeddingClassifier, self).__init__()

        self.activation = parameters["linear"]["activation"]
        self.n_classes  = parameters["n_classes"]

        self.linear_blocks = torch.nn.ModuleList([LinearBlock(parameters["linear"]["in_dim"][i],
                                                 parameters["linear"]["out_dim"][i],
                                                 parameters["linear"]["activation"])
                                                 for i in range(0, len(parameters["linear"]["in_dim"]))])

        self.output = torch.nn.Linear(parameters["output"]["in_dim"],
                                      parameters["output"]["out_dim"])

        if load_checkpoint:
            self.load_from_checkpoint(parameters)

    def forward(self, X, dropout_p=None):

        # Feed forward input
        for block in self.linear_blocks:
            X = block(X, dropout_p)

        # Output prediction as logits
        return self.output(X)

    def load_from_checkpoint(self, parameters):
        classifier_weights = torch.load(parameters["checkpoint"],
                                        map_location=parameters["map_location"])

        keys       = classifier_weights["state_dict"].keys()
        new_keys   = [".".join(k.split(".")[1:]) for k in keys]
        state_dict = {nk: classifier_weights["state_dict"][k] for (k, nk) in zip(keys, new_keys)}

        self.load_state_dict(state_dict)
        return self

class AdversarialClassifier(torch.nn.Module):

    def __init__(self, parameters):

        super(AdversarialClassifier, self).__init__()

        self.n_classes = parameters["n_classes"]

        self.input_normalization = torch.nn.BatchNorm1d(parameters["linear"]["in_dim"][0])
        self.linear_blocks = torch.nn.ModuleList([LinearBlock(parameters["linear"]["in_dim"][i],
                                                  parameters["linear"]["out_dim"][i],
                                                  parameters["linear"]["activation"])
                                                  for i in range(0, len(parameters["linear"]["in_dim"]))])

        self.output = torch.nn.Linear(parameters["output"]["in_dim"], parameters["output"]["out_dim"])
        self.grl = GradientReversalLayer()

    def forward(self, X, dropout_p=None):

        # Run first through Gradient Reversal Layer
        X = self.grl(X)

        # Instance normalization
        X = self.input_normalization(X)

        # Run model
        for block in self.linear_blocks:
            X = block(X, dropout_p)

        # Output prediction as logits
        return self.output(X)
