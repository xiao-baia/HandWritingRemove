import torch
import torch.nn as nn
import torch.nn.functional as F
from model.GeoTr import GeoTr

def reload_model(model, path, device):
    if not bool(path):
        return model
    else:
        model_dict = model.state_dict()
        # pretrained_dict = torch.load(path, map_location='cuda:0')
        pretrained_dict = torch.load(path, map_location=device)
        # print(len(pretrained_dict.keys()))
        # print(len(pretrained_dict.keys()))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        return model

class GeoTrP(nn.Module):
    def __init__(self):
        super(GeoTrP, self).__init__()
        self.GeoTr = GeoTr()

    def forward(self, x):
        bm = self.GeoTr(x)  # [0]
        bm = 2 * (bm / 288) - 1

        bm = (bm + 1) / 2 * 2560

        bm = F.interpolate(bm, size=(2560, 2560), mode='bilinear', align_corners=True)

        return bm

def DocTr_Plus(weights, device):
    _GeoTrP = GeoTrP()
    _GeoTrP = _GeoTrP.to(device)
    reload_model(_GeoTrP.GeoTr, weights, device)
    _GeoTrP.eval()
    return _GeoTrP