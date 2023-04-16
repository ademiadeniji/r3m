# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
from numpy.core.numeric import full
import torch
import torch.nn as nn
from torch.nn.modules.activation import Sigmoid
from torch.nn.modules.linear import Identity
import torchvision
from torchvision import transforms
from r3m import utils
from pathlib import Path
from torchvision.utils import save_image
import torchvision.transforms as T

epsilon = 1e-8
def do_nothing(x): return x

class R3M(nn.Module):
    def __init__(self, device, lr, hidden_dim, size=34, l2weight=1.0, l1weight=1.0, 
                 langweight=1.0, tcnweight=0.0, l2dist=True, bs=16, linear_probe=False,
                 finetune_lang_reward=False, finetune_bn_layers=False, freeze_convnet=False,
                 freeze_lang_reward=False):
        super().__init__()

        self.device = device
        self.use_tb = False
        self.l2weight = l2weight
        self.l1weight = l1weight
        self.tcnweight = tcnweight ## Weight on TCN loss (states closer in same clip closer in embedding)
        self.l2dist = l2dist ## Use -l2 or cosine sim
        self.langweight = langweight ## Weight on language reward
        self.size = size ## Size ResNet or ViT
        self.num_negatives = 3
        self.linear_probe = linear_probe
        self.finetune_lang_reward = finetune_lang_reward
        self.finetune_bn_layers = finetune_bn_layers
        self.freeze_convnet = freeze_convnet
        self.freeze_lang_reward = freeze_lang_reward

        ## Distances and Metrics
        self.cs = torch.nn.CosineSimilarity(1)
        self.bce = nn.BCELoss(reduce=False)
        self.sigm = Sigmoid()

        params = []
        ######################################################################## Sub Modules
        ## Visual Encoder
        if size == 18:
            self.outdim = 512
            self.convnet = torchvision.models.resnet18(pretrained=False)
        elif size == 34:
            self.outdim = 512
            self.convnet = torchvision.models.resnet34(pretrained=False)
        elif size == 50:
            self.outdim = 2048
            self.convnet = torchvision.models.resnet50(pretrained=False)
        elif size == 0:
            from transformers import AutoConfig
            self.outdim = 768
            self.convnet = AutoModel.from_config(config = AutoConfig.from_pretrained('google/vit-base-patch32-224-in21k')).to(self.device)

        if self.size == 0:
            self.normlayer = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        else:
            self.normlayer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.convnet.fc = Identity()
        self.convnet.train()

        if self.finetune_bn_layers:
            self.convnet.requires_grad_(False)
            for name, param in self.convnet.named_parameters():
                if 'bn' in name:
                    param.requires_grad_(True)
                    if 'weight' in name:
                        nn.init.ones_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
                    else:
                        raise ValueError('Error in BN reset')
                    params += [param]
        elif self.freeze_convnet:
            self.convnet.requires_grad_(False)
        else:
            params += list(self.convnet.parameters())

        ## Language Reward
        if self.langweight > 0.0:
            ## Pretrained DistilBERT Sentence Encoder
            from r3m.models.models_language import LangEncoder, LanguageReward
            self.lang_enc = LangEncoder(self.device, 0, 0) 
            self.lang_rew = LanguageReward(None, self.outdim, hidden_dim, self.lang_enc.lang_size, simfunc=self.sim) 
            if self.linear_probe:
                self.lang_rew.requires_grad_(False)
                self.lang_rew.pred[8].requires_grad_(True)
                self.lang_rew.pred[8].reset_parameters()
                for name, param in self.lang_rew.named_parameters():
                    if name == 'pred.8.weight' or name == 'pred.8.bias':
                        params += [param]
            elif self.finetune_lang_reward:
                for linear_idx in range(len(self.lang_rew.pred)):
                    try:
                        self.lang_rew.pred[linear_idx].reset_parameters()
                    except:
                        pass
                params += list(self.lang_rew.parameters())
            elif self.freeze_lang_reward:
                self.lang_rew.requires_grad_(False)
            else:
                params += list(self.lang_rew.parameters())
        ########################################################################

        ## Optimizer
        self.encoder_opt = torch.optim.Adam(params, lr = lr)

    def get_lang_enc(self, sentences):
        return self.lang_enc(sentences)
    
    def get_reward_le(self, e0, es, le):
        return self.lang_rew(e0, es, le)

    def get_reward(self, e0, es, sentences):
        ## Only callable is langweight was set to be 1
        le = self.lang_enc(sentences)
        return self.lang_rew(e0, es, le)

    ## Forward Call (im --> representation)
    def forward(self, obs, num_ims = 1, obs_shape = [3, 224, 224]):
        if obs_shape != [3, 224, 224]:
            preprocess = nn.Sequential(
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        self.normlayer,
                )
        else:
            preprocess = nn.Sequential(
                        self.normlayer,
                )

        ## Input must be [0, 255], [3,244,244]
        obs = obs.float() /  255.0
        obs_p = preprocess(obs)
        h = self.convnet(obs_p)
        return h

    def sim(self, tensor1, tensor2):
        if self.l2dist:
            d = - torch.linalg.norm(tensor1 - tensor2, dim = -1)
        else:
            d = self.cs(tensor1, tensor2)
        return d
