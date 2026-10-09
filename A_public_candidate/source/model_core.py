"""Six- and eleven-layer models used by the current MNIST pipeline.

Definitions are extracted unchanged from DeepBMNN_model1_v5_layer_11.py.
"""

import torch.nn as nn

from DeepBMNN_modules1_v5 import MLUTx0, MLUT_prophase_oddLinear, MLUT_prophase_evenLinear
from DeepBMNN_modules2_v5 import (
    MLUT_switch_Linear2,
    MLUT_anaphase_Linear1_2, MLUT_anaphase_Linear2_2,
    MLUT_anaphase_Linear3_2, MLUT_anaphase_Linear4_2,
    MLUT_anaphase_Linear5_2, MLUT_anaphase_Linear6,
)
from DeepBMNN_modules2_non_binary import MLUTx0_48, MLUT_input_Linear1_b


class DeepBMNN_6_layer_reverse_binary(nn.Module):#MNN(拡張なし)
    def __init__(self):
        super(DeepBMNN_6_layer_reverse_binary,self).__init__()

        self.MLUTx0_pro = MLUTx0_48(784,48)#MLUT外部

        self.MLUTx1_pro = MLUT_input_Linear1_b(48,40)
        self.MLUTx2_pro = MLUT_anaphase_Linear1_2(40,32)
        self.MLUTx3_pro = MLUT_anaphase_Linear2_2(32,24)
        self.MLUTx4_pro = MLUT_anaphase_Linear3_2(24,16)
        self.MLUTx5_pro = MLUT_anaphase_Linear4_2(16,8)
        self.MLUTx6_pro = MLUT_anaphase_Linear5_2(8,8)
        
        self.out_MLUT_pro = MLUT_anaphase_Linear6(8,10)#MLUT外部

    def forward(self,x):
        x = x.view(x.size(0),-1)

        MLUTx0y0_outpro,MLUTx0y1_outpro,MLUTx0y2_outpro,MLUTx0y3_outpro,MLUTx0y4_outpro,MLUTx0y5_outpro = self.MLUTx0_pro(x)
        
        MLUTx1y0_outpro,MLUTx1y1_outpro,MLUTx1y2_outpro,MLUTx1y3_outpro,MLUTx1y4_outpro,MLUTx1y5_outpro = self.MLUTx1_pro(MLUTx0y0_outpro,MLUTx0y1_outpro,MLUTx0y2_outpro,MLUTx0y3_outpro,MLUTx0y4_outpro,MLUTx0y5_outpro)
        MLUTx2y0_outpro,MLUTx2y1_outpro,MLUTx2y2_outpro,MLUTx2y3_outpro,MLUTx2y4_outpro = self.MLUTx2_pro(MLUTx1y0_outpro,MLUTx1y1_outpro,MLUTx1y2_outpro,MLUTx1y3_outpro,MLUTx1y4_outpro,MLUTx1y5_outpro)
        MLUTx3y1_outpro,MLUTx3y2_outpro,MLUTx3y3_outpro,MLUTx3y4_outpro = self.MLUTx3_pro(MLUTx2y0_outpro,MLUTx2y1_outpro,MLUTx2y2_outpro,MLUTx2y3_outpro,MLUTx2y4_outpro)
        MLUTx4y1_outpro,MLUTx4y2_outpro,MLUTx4y3_outpro = self.MLUTx4_pro(MLUTx3y1_outpro,MLUTx3y2_outpro,MLUTx3y3_outpro,MLUTx3y4_outpro)
        MLUTx5y2_outpro,MLUTx5y3_outpro = self.MLUTx5_pro(MLUTx4y1_outpro,MLUTx4y2_outpro,MLUTx4y3_outpro)
        MLUTx6y2_outpro = self.MLUTx6_pro(MLUTx5y2_outpro,MLUTx5y3_outpro)

        MLUT_out = self.out_MLUT_pro(MLUTx6y2_outpro)
        return MLUT_out    


class DeepBMNN_11_layer(nn.Module):#二値化BMNN
    def __init__(self):
        super(DeepBMNN_11_layer,self).__init__()

        self.MLUTx0_pro=MLUTx0(784,44)
        self.MLUTx1_pro = MLUT_prophase_oddLinear(44,44)
        self.MLUTx2_pro = MLUT_prophase_evenLinear(44,44)
        self.MLUTx3_pro = MLUT_prophase_oddLinear(44,44)
        self.MLUTx4_pro = MLUT_prophase_evenLinear(44,44)
        self.MLUTx5_pro = MLUT_prophase_oddLinear(44,44)
        
        self.MLUTx6_pro = MLUT_switch_Linear2(44,40)

        self.MLUTx5_ana = MLUT_anaphase_Linear1_2(40,32)
        self.MLUTx4_ana = MLUT_anaphase_Linear2_2(32,24)
        self.MLUTx3_ana = MLUT_anaphase_Linear3_2(24,16)
        self.MLUTx2_ana = MLUT_anaphase_Linear4_2(16,8)

        self.MLUTx1_ana = MLUT_anaphase_Linear5_2(8,8)
        
        self.MLUTx0_ana = MLUT_anaphase_Linear6(8,10)

    def forward(self,x):
        x = x.view(x.size(0),-1)

        MLUTx0y0_outpro,MLUTx0y1_outpro,MLUTx0y2_outpro,MLUTx0y3_outpro,MLUTx0y4_outpro,MLUTx0y5_outpro = self.MLUTx0_pro(x)
        MLUTx1y0_outpro,MLUTx1y1_outpro,MLUTx1y2_outpro,MLUTx1y3_outpro,MLUTx1y4_outpro,MLUTx1y5_outpro = self.MLUTx1_pro(MLUTx0y0_outpro,MLUTx0y1_outpro,MLUTx0y2_outpro,MLUTx0y3_outpro,MLUTx0y4_outpro,MLUTx0y5_outpro)
        MLUTx2y0_outpro,MLUTx2y1_outpro,MLUTx2y2_outpro,MLUTx2y3_outpro,MLUTx2y4_outpro,MLUTx2y5_outpro = self.MLUTx2_pro(MLUTx1y0_outpro,MLUTx1y1_outpro,MLUTx1y2_outpro,MLUTx1y3_outpro,MLUTx1y4_outpro,MLUTx1y5_outpro)
        MLUTx3y0_outpro,MLUTx3y1_outpro,MLUTx3y2_outpro,MLUTx3y3_outpro,MLUTx3y4_outpro,MLUTx3y5_outpro = self.MLUTx3_pro(MLUTx2y0_outpro,MLUTx2y1_outpro,MLUTx2y2_outpro,MLUTx2y3_outpro,MLUTx2y4_outpro,MLUTx2y5_outpro)
        MLUTx4y0_outpro,MLUTx4y1_outpro,MLUTx4y2_outpro,MLUTx4y3_outpro,MLUTx4y4_outpro,MLUTx4y5_outpro = self.MLUTx4_pro(MLUTx3y0_outpro,MLUTx3y1_outpro,MLUTx3y2_outpro,MLUTx3y3_outpro,MLUTx3y4_outpro,MLUTx3y5_outpro)
        MLUTx5y0_outpro,MLUTx5y1_outpro,MLUTx5y2_outpro,MLUTx5y3_outpro,MLUTx5y4_outpro,MLUTx5y5_outpro = self.MLUTx5_pro(MLUTx4y0_outpro,MLUTx4y1_outpro,MLUTx4y2_outpro,MLUTx4y3_outpro,MLUTx4y4_outpro,MLUTx4y5_outpro)


        MLUTx6y0_outpro,MLUTx6y1_outpro,MLUTx6y2_outpro,MLUTx6y3_outpro,MLUTx6y4_outpro,MLUTx6y5_outpro = self.MLUTx6_pro(MLUTx5y0_outpro,MLUTx5y1_outpro,MLUTx5y2_outpro,MLUTx5y3_outpro,MLUTx5y4_outpro,MLUTx5y5_outpro)


        MLUTx5y0_outana,MLUTx5y1_outana,MLUTx5y2_outana,MLUTx5y3_outana,MLUTx5y4_outana = self.MLUTx5_ana(MLUTx6y0_outpro,MLUTx6y1_outpro,MLUTx6y2_outpro,MLUTx6y3_outpro,MLUTx6y4_outpro,MLUTx6y5_outpro)
        MLUTx4y1_outana,MLUTx4y2_outana,MLUTx4y3_outana,MLUTx4y4_outana = self.MLUTx4_ana(MLUTx5y0_outana,MLUTx5y1_outana,MLUTx5y2_outana,MLUTx5y3_outana,MLUTx5y4_outana)
        MLUTx3y2_outana,MLUTx3y3_outana,MLUTx3y4_outana = self.MLUTx3_ana(MLUTx4y1_outana,MLUTx4y2_outana,MLUTx4y3_outana,MLUTx4y4_outana)
        MLUTx2y2_outana,MLUTx2y3_outana = self.MLUTx2_ana(MLUTx3y2_outana,MLUTx3y3_outana,MLUTx3y4_outana)
        MLUTx1y2_outana = self.MLUTx1_ana(MLUTx2y2_outana,MLUTx2y3_outana)

        MLUTx0y2_outana = self.MLUTx0_ana(MLUTx1y2_outana)

        return MLUTx0y2_outana
