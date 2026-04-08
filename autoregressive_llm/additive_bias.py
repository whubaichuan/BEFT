import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import torch
from torch import nn
from torch.nn import functional as F
import math

def find_module(root_module: nn.Module, key: str):
    sub_keys = key.split(".")
    parent_module = root_module
    for sub_key in sub_keys[:-1]:
        parent_module = getattr(parent_module, sub_key)
    module = getattr(parent_module, sub_keys[-1])
    return parent_module, sub_keys[-1], module


class AddbiasLinear(nn.Linear):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        self.add_bias = nn.Parameter(self.weight.new_zeros((out_features)))
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        # if self.add_bias is not None:
        #     nn.init.zeros_(self.add_bias)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
      
    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        result = F.linear(x, T(self.weight), bias=self.bias)
        result += self.add_bias
        return result


class Addbias:

    def __init__(self, model, float16):
        self.model = model
        self.hidden_dim = model.config.hidden_size
        self.float16 = float16
        self.change_bias = 'value' # 'query', 'key', 'value'
        print(model.config.model_type)
        if model.config.model_type == "llama":
            attention_name = "self_attn"
        elif model.config.model_type == "gptj":
            attention_name = "attn"
        else:
            raise NotImplementedError

        # add bias
        for key, _ in model.named_modules():
            if key[-len(attention_name):] == attention_name:
                logger.info(f"add bias to: {key}")
                _, _, attn = find_module(model, key)

                if model.config.model_type == "llama":
                    if self.change_bias == 'query':
                        original_q_weight = attn.q_proj.weight.data
                        attn.q_proj = AddbiasLinear(model.config.hidden_size, model.config.hidden_size, bias=False).to(original_q_weight.device)
                        if float16:
                            attn.q_proj.half()
                        attn.q_proj.weight.data = original_q_weight 
                    elif self.change_bias == 'value':
                        original_v_weight= attn.v_proj.weight.data
                        attn.v_proj = AddbiasLinear(model.config.hidden_size, model.config.hidden_size, bias=False).to(original_v_weight.device)
                        if float16:
                            attn.v_proj.half()
                        attn.v_proj.weight.data = original_v_weight
                    elif self.change_bias == 'key':
                        original_k_weight= attn.k_proj.weight.data
                        attn.k_proj = AddbiasLinear(model.config.hidden_size, model.config.hidden_size, bias=False).to(original_k_weight.device)
                        if float16:
                            attn.k_proj.half()
                        attn.k_proj.weight.data = original_k_weight
                elif model.config.model_type == "gptj":
                    if self.change_bias == 'value':
                        original_v_weight= attn.v_proj.weight.data
                        attn.v_proj = AddbiasLinear(model.config.hidden_size, model.config.hidden_size, bias=False).to(original_v_weight.device)
                        if float16:
                            attn.v_proj.half()
                        attn.v_proj.weight.data = original_v_weight
                    elif self.change_bias == 'key':
                        original_k_weight= attn.k_proj.weight.data
                        attn.k_proj = AddbiasLinear(model.config.hidden_size, model.config.hidden_size, bias=False).to(original_k_weight.device)
                        if float16:
                            attn.k_proj.half()
                        attn.k_proj.weight.data = original_k_weight
                    elif self.change_bias == 'query':
                        original_q_weight = attn.q_proj.weight.data
                        attn.q_proj = AddbiasLinear(model.config.hidden_size, model.config.hidden_size, bias=False).to(original_q_weight.device)
                        if float16:
                            attn.q_proj.half()
                        attn.q_proj.weight.data = original_q_weight
                else:
                    raise NotImplementedError
        
        # Freeze non-Addbias parameters
        for n, p in model.named_parameters():
            if "add_bias" not in n:
                p.requires_grad = False


