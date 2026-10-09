######################################binarized_modules#################################################
# This code is written by XihongZhou (PhD student)
# date:2020-12-16
# E-mail: g863003a@mails.cc.ehime-u.ac.jp
# Address: Takahashi Lab, Computer System Laboratory, Department of Computer Science, Ehime University
########################################################################################################

import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 让torch判断是否使用GPU，建议使用GPU环境，因为会快很多

# def Binarize(tensor,quant_mode='det'):
#     if quant_mode=='det':
#         # tensor=torch.tensor(tensor,dtype=torch.float)
#         return tensor.sign()
#     else:
#         return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0,1).round().mul_(2).add_(-1)

def Binarize(tensor):    # -> -1 or 1
    return torch.where(tensor <= 0, -1, 1).clone().detach().to(dtype=torch.float)



# class BinarizeLinear(nn.Linear):

#     def __init__(self, *kargs, **kwargs):
#         super(BinarizeLinear, self).__init__(*kargs, **kwargs)

#     def forward(self, input):
#         if input.size(1) != 784:
#             input.data=Binarize(input.data)
#         if not hasattr(self.weight,'org'):
#             self.weight.org=self.weight.data.clone()
#         self.weight.data=Binarize(self.weight.org)
#         out = nn.functional.linear(input, self.weight)
#         if not self.bias is None:
#             self.bias.org=self.bias.data.clone()
#             out += self.bias.view(1, -1).expand_as(out)
#         return out
#二値化線形NN層
class BinarizeLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)
        #nn.linearのinit
    def forward(self, input):
        if input.size(1) != 784:
            input.data=Binarize(input.data)
        out = nn.functional.linear(input, self.weight)
        if not self.bias is None:#バイアスが存在suru場合に出力を追加
            out += self.bias.view(1, -1).expand_as(out)
        return out
    

class Linears(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(Linears, self).__init__(*kargs, **kwargs)
        #nn.linearのinit
    def forward(self, input):
        out = nn.functional.linear(input, self.weight)
        if not self.bias is None:#バイアスが存在suru場合に出力を追加
            out += self.bias.view(1, -1).expand_as(out)
        return out 