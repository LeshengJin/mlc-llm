"""This file specifies how MLC's Llama parameters are quantized using group quantization
or other formats."""
from typing import Tuple

from tvm.relax.frontend import nn

from ...loader import QuantizeMapping
from ...quantization import GroupQuantize
from .gpt_neox_model import GPTNeoXConfig, GPTNeoXForCausalLM


def group_quant(
    model_config: GPTNeoXConfig,
    quantization: GroupQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a Llama-architecture model using group quantization."""
    model: nn.Module = GPTNeoXForCausalLM(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    model = quantization.quantize_model(
        model,
        quant_map,
        "",
    )
    return model, quant_map
