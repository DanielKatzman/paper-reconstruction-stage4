"""New capacities do not alter the historical 27-model catalogue or its weights."""
from dataclasses import replace
from torch import nn

from .model_catalogue import catalogue
from .profile_models import PairModel, SequenceEncoder
MODELS = ("pair_mlp_small", "paired_strip_cnn_small")
BASE_MODELS = {"pair_mlp_small": "pair_mlp", "paired_strip_cnn_small": "paired_strip_cnn"}
from .contracts import require


class RerankerModel(PairModel):
    def __init__(self, name, config, trial):
        require(name in MODELS, 'Unknown v0.7 reranker')
        spec = catalogue()[BASE_MODELS[name]]
        super().__init__(spec, config, trial)
        if self.kind == 'P3': return
        medium = name.endswith('_medium'); width = 256 if medium else 128
        self.geometry = SequenceEncoder(replace(spec, name='tcn_small', width=width, layers=4 if medium else 2),
                                        trial['bundle'], self.checkpointed)
        dimension = width*4+8
        if self.kind == 'P2':
            channels = (64, 128, 256) if medium else (32, 64, 128)
            modules = []; previous = 8
            for i, channel in enumerate(channels):
                modules.extend((nn.Conv2d(previous, channel, 3, stride=1 if i == 0 else 2, padding=1),
                                nn.GroupNorm(8, channel), nn.GELU())); previous = channel
            self.cnn = nn.Sequential(*modules, nn.AdaptiveAvgPool2d(1)); dimension += channels[-1]
        hidden = 256 if medium else 128
        self.score = nn.Sequential(nn.Linear(dimension, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, a, b, evidence, reverse_evidence=None):
        first = super().forward(a, b, evidence)
        if self.kind == 'P3': return first
        require(reverse_evidence is not None, 'Symmetric pair scoring requires reverse observable fit evidence')
        second = super().forward(b, a, reverse_evidence)
        return {'compatibility': (first['compatibility']+second['compatibility'])/2}

    def encode_geometry(self, sequence):
        import torch
        x = self.geometry(sequence)
        return torch.cat((x.mean(1), x.amax(1)), 1)

    def compatibility_from_geometry(self, a, b, px, py, evidence, reverse_evidence):
        import torch
        require(not self.training and self.kind in ('P1', 'P2'), 'Geometry reuse is inference only')
        common = (torch.abs(px-py), px*py)
        def score(first, second, e):
            features = torch.cat((*common, e), 1)
            if self.kind == 'P2':
                image = torch.cat((first['strip'], torch.flip(second['strip'], [2])), 1)
                other = torch.cat((image[:,4:], image[:,:4]), 1)
                vision = (self.cnn(image)+self.cnn(other)).flatten(1)/2
                features = torch.cat((features, vision), 1)
            return self.score(features).squeeze(-1)
        return (score(a, b, evidence)+score(b, a, reverse_evidence))/2

    def compatibility_only(self, a, b, evidence, reverse_evidence):
        import torch
        require(not self.training and self.kind == 'P3', 'Attention compatibility fast path is inference only')
        x, y = self.stem(a['sequence']), self.stem(b['sequence'])
        for layer in self.blocks:
            xx = layer(x, y); y = layer(y, x); x = xx
        px = torch.cat((x.mean(1), x.amax(1)), 1); py = torch.cat((y.mean(1), y.amax(1)), 1)
        return self.score(torch.cat((torch.abs(px-py), px*py), 1)).squeeze(-1)
