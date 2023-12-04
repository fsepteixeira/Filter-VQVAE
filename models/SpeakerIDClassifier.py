
import torch
import torch.nn.functional as F

class SpeakerIDClassifier(torch.nn.Module):

    def __init__(self, parameters):
        super(SpeakerIDClassifier, self).__init__()
        self.weight = torch.nn.Parameter(torch.FloatTensor(parameters["n_speakers"], parameters["n_features"]))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, X):
        return F.linear(F.normalize(X), F.normalize(self.weight))

    @staticmethod
    def load_from_checkpoint(parameters):
        classifier = SpeakerIDClassifier(parameters)
        state_dict = torch.load(parameters["source"])["state_dict"]
        state_dict["weight"] = state_dict["id_classifier.weight"]
        del state_dict["id_classifier.weight"]
        classifier.load_state_dict(state_dict)
        return classifier

