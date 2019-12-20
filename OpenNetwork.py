import torch
from Network import network

nw = torch.load(network, "C:/Projects/PyTorch_tutorial/model.pt")
nw.eval()