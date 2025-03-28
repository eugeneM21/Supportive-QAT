from quantize.quantizer import UniformAffineQuantizer, RoundUpQuantizer, RoundDownQuantizer
import torch.nn as nn

class QuantLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, quantizer_cls=UniformAffineQuantizer):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        self.quantizer = quantizer_cls(weight=self.linear.weight, group_size=self.linear.weight.shape[-1])

    def forward(self, x):
        quantized_weight = self.quantizer(self.linear.weight)
        quantized_weight = quantized_weight.to(dtype=x.dtype, device=x.device)
        return nn.functional.linear(x, quantized_weight, self.linear.bias)


def replace_linear_layers(model, quantizer_cls):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            in_features = module.in_features
            out_features = module.out_features
            bias = module.bias is not None
            setattr(model, name, QuantLinear(in_features, out_features, bias, quantizer_cls))
        else:
            replace_linear_layers(module, quantizer_cls)
    return model
