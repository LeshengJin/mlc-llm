"""
This file specifies how MLC's GPT parameter maps from other formats, for example HuggingFace
PyTorch, HuggingFace safetensors.
"""
import functools

from ...loader import ExternMapping
from ...quantization import Quantization
from .gpt_neox_model import GPTNeoXConfig, GPTNeoXForCausalLM


def huggingface(model_config: GPTNeoXConfig, quantization: Quantization) -> ExternMapping:
    """Returns a parameter mapping that maps from the names of MLC LLM parameters to
    the names of HuggingFace PyTorch parameters.

    Parameters
    ----------
    model_config : GPTNeoXConfig
        The configuration of the GPTNeoX model.

    quantization : Quantization
        The quantization configuration.

    Returns
    -------
    param_map : ExternMapping
        The parameter mapping from MLC to HuggingFace PyTorch.
    """
    model = GPTNeoXForCausalLM(model_config)
    if quantization is not None:
        model.to(quantization.model_dtype)
    _, _named_params = model.export_tvm(spec=model.get_default_spec())
    named_parameters = dict(_named_params)

    mapping = ExternMapping()

    for i in range(model_config.num_hidden_layers):
        # inv_freq is not used in the model
        attn = f"gpt_neox.layers.{i}.attention"
        mapping.add_unused(f"{attn}.rotary_emb.inv_freq")
        mapping.add_unused(f"{attn}.masked_bias")
        mapping.add_unused(f"{attn}.bias")

    for mlc_name, mlc_param in named_parameters.items():
        if "layernorm" in mlc_name or "layer_norm" in mlc_name or "embed_out" in mlc_name:
            param_dtype = "float32"
        elif ".dense_h_to_4h.bias" in mlc_name or ".dense_4h_to_h.bias" in mlc_name:
            param_dtype = model_config.ffn_out_dtype
        else:
            param_dtype = mlc_param.dtype
        mapping.add_mapping(
            mlc_name,
            [mlc_name],
            functools.partial(
                lambda x, dtype: x.astype(dtype),
                dtype=param_dtype,
            ),
        )
    return mapping
