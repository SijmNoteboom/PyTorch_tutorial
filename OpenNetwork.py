import torch
import pickle

# model = torch.load(model, map_location="/home/cdac-user/PycharmProjects/PyTorch_tutorial/model.pt")
vocab = Network(*args, **kwargs)
vocab.load_state_dict(torch.load("/home/cdac-user/PycharmProjects/PyTorch_tutorial/vocab.pt"))
vocab.eval()
