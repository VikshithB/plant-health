# train_debug.py
import torch, torchvision.models as M
print("step 1 – import done")

net = M.resnet50(weights=M.ResNet50_Weights.DEFAULT)
print("step 2 – model instantiated")

print("device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
