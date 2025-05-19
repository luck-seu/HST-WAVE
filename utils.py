import torch
import numpy as np
def get_large_label_hz():
    indexs = [1,15,24,39,50,76,89,93]
    return torch.LongTensor(indexs)

def get_large_label_jh():
    indexs = [12,22,26,29,41,44,52,68,73,80,87,97,103]
    return torch.LongTensor(indexs)

def get_large_label_test_hz():
    indexs = [84,1,24,15,50,74,39,93,89,76]
    return torch.LongTensor(indexs)

def get_large_label_test_jh():
    indexs = [129,97,92,12,26,41,68,80,87,73]
    return torch.LongTensor(indexs)

def get_graph_dict(data):
    if data=='jh':
        dict = torch.load('data/JinHua/graph_info.pt')
    elif data=='hz':
        dict = torch.load('data/HangZ/graph_info.pt')
    else:
        print("Error choice in graph loading!")
        return {}
    return dict


