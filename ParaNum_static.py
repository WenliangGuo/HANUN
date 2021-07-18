from thop import profile
from model import AUPSNet
from model import u2net
import matplotlib.pyplot as plt
import numpy as np

def params_count(model):
    """
    Compute the number of parameters.
    Args:
        model (model): model to count the number of parameters.
    """
    return np.sum([p.numel() for p in model.parameters()]).item()

def plot_bar():

    xx = ('AUPSNet', 'AUPSNet-5', 'AUPSNet-7', '-allPP', '-noSE', 'U2Net')
    yy = paranum 

    plt.figure(figsize=(12, 6))
    plt.rcParams.update({'font.size': 18})

    plt.bar(xx, yy)
    plt.ylabel("Number of Parameters (Millions)")

    plt.savefig("paranum_statistic.png")
    plt.show()

model = []
model.append( AUPSNet.AUPSNET(3,1))
model.append( AUPSNet.AUPSNET_5(3,1))
model.append( AUPSNet.AUPSNET_7(3,1))
model.append( AUPSNet.AUPSNET_ALLPP(3,1))
model.append( AUPSNet.AUPSNET_NOSE(3,1))
model.append( u2net.U2NET(3,1))

paranum = []

for i in range(len(model)):
    num = params_count(model[i])
    print(num)
    num = num * 1.0 / 1000000
    paranum.append(num)

plot_bar()
#flops, params = profile(model, inputs=(input,))
