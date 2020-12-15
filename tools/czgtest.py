import torch
pthfile = '../czg_log/fpn_selfatt_4convhead_gamma-1/epoch_11.pth'
net = torch.load(pthfile)

# 'neck.fpn_selfatt.0.value_conv.weight'
for key,value in net["state_dict"].items():
    # if 'gamma' in key:
    print(key,value.size(), sep=" ")