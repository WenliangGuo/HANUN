from thop import profile
from model import UPSPNet
import numpy as np

def params_count(model):
    """
    Compute the number of parameters.
    Args:
        model (model): model to count the number of parameters.
    """
    return np.sum([p.numel() for p in model.parameters()]).item()

model = []
model.append( UPSPNet.UPSPNET_4(3,1))
model.append(UPSPNet.UPSPNET_5(3,1))
model.append( UPSPNet.UPSPNET(3,1))
model.append( UPSPNet.UPSPNET_7(3,1))
model.append( UPSPNet.UPSPNET_8(3,1))

for i in range(5):
    print("depth={}: ".format(str(i+4)),params_count(model[i]))
#flops, params = profile(model, inputs=(input,))
