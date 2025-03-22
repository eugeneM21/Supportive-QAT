from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from quantize.quantizer import UniformAffineQuantizer, RoundUpQuantizer, RoundDownQuantizer


class QuantLinear(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    class QuantType(Enum):
        REGULAR = 0
        UP = 1
        DOWN = 1

    def __init__(
        self,
        org_module: nn.Linear,
        wbits=4,
        group_size=64,
        quant_type = QuantType.REGULAR
    ):
        super().__init__()
        self.fwd_kwargs = dict()
        self.fwd_func = F.linear
        self.register_parameter('weight',org_module.weight) # trainable
        if org_module.bias is not None:
            self.register_buffer('bias',org_module.bias)
        else:
            self.bias = None
        self.in_features = org_module.in_features
        self.out_features = org_module.out_features
        # de-activate the quantized forward default
        self.use_weight_quant = False
        # initialize quantizer
        self.weight_quantizer = None
        
        if quant_type == self.QuantType.REGULAR:
            self.weight_quantizer = UniformAffineQuantizer(wbits, group_size, weight=org_module.weight)
        elif quant_type == self.QuantType.UP:
            self.weight_quantizer = RoundUpQuantizer(wbits, group_size, weight=org_module.weight)
        elif quant_type == self.QuantType.DOWN:
            self.weight_quantizer = RoundDownQuantizer(wbits, group_size, weight=org_module.weight)

        assert self.weight_quantizer is not None, "Quantizer is not initialized"
        self.use_temporary_parameter = False

    
    
    def forward(self, input: torch.Tensor):
        if self.use_weight_quant:
            weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.weight
            bias = self.bias

        
        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)


        return out

    def set_quant_state(self, weight_quant: bool = False):
        self.use_weight_quant = weight_quant




