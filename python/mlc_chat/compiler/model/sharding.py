# pylint: disable=invalid-name
"""This file creates shard functions that split model parameters into N shards for
tensor parallelsim multi-gpu inference."""
from typing import List, Tuple

from tvm import te, tir, topi
from tvm.relax.frontend import nn

from .model import ModelConfig


def create_shard_func(
    name: str, param: nn.Parameter, num_shards: int, model_config: ModelConfig
) -> Tuple[str, tir.PrimFunc, Tuple[List[int], str]]:
    """
    Create shard functions that split the parameters into `num_shards`
    for tensor parallelsim multi-gpu inference.
    Parameters
    ----------
    name : str
        Parameter name.
    param : nn.Parameter
        The parameter to be sharded.
    num_shards : int
        The number of shards.
    model_config : ModelConfig
        The model configuration.
    Returns
    -------
    shard_func_name : str
        The name of the shard function.
    shard_func : tir.PrimFunc
        The shard function.
    shape_dtype : Tuple[List[int], str]
        The shape and dtype of the output of the shard function.
    """
    if param.shard_strategy is None or num_shards == 1:
        return None, None, None
    layer_name, weight_name = name.split(".")[-2], name.split(".")[-1]
    if param.shard_strategy == "shard_qkv":
        shard_func_name = f"shard_qkv_{layer_name}_{weight_name}"
        shard_func = _shard_qkv_weight_scale(param, num_shards, model_config)
    elif param.shard_strategy == "shard_linear_column":
        shard_func_name = f"shard_linear_column_{layer_name}_{weight_name}"
        shard_func = _shard_linear_column_weight_scale(param, num_shards, model_config)
    elif param.shard_strategy == "shard_gate_up":
        shard_func_name = f"shard_gate_up_{layer_name}_{weight_name}"
        shard_func = _shard_gate_up_weight_scale(param, num_shards, model_config)
    else:
        raise NotImplementedError(f"Shard strategy not implemented: {param.shard_strategy}")
    buffer = shard_func.buffer_map[shard_func.params[-1]]
    shape = [int(i) for i in buffer.shape]
    dtype = str(buffer.dtype)
    return shard_func_name, shard_func, [shape, dtype]


def _shard_qkv_weight_scale(
    weight: nn.Parameter, num_shards: int, model_config: ModelConfig
) -> tir.PrimFunc:
    head_dim = model_config.head_dim
    q_heads = model_config.num_attention_heads
    kv_heads = model_config.num_key_value_heads
    (spatial, red), dtype = weight.shape, weight.dtype
    spatial, red = int(spatial) * num_shards, int(red)
    a = te.placeholder((spatial, red), dtype=dtype)
    w = topi.reshape(a, (spatial // head_dim, head_dim, red))
    q = te.compute((q_heads, head_dim, red), lambda i, j, k: w[i, j, k])
    k = te.compute((kv_heads, head_dim, red), lambda i, j, k: w[q_heads + i, j, k])
    v = te.compute((kv_heads, head_dim, red), lambda i, j, k: w[q_heads + kv_heads + i, j, k])
    q = topi.reshape(q, (num_shards, q_heads // num_shards, head_dim, red))
    k = topi.reshape(k, (num_shards, kv_heads // num_shards, head_dim, red))
    v = topi.reshape(v, (num_shards, kv_heads // num_shards, head_dim, red))
    w = topi.concatenate((q, k, v), axis=1)
    w = topi.reshape(w, (num_shards, (q_heads + kv_heads * 2) // num_shards * head_dim, red))
    func = te.create_prim_func([a, w])
    return func


def _shard_linear_column_weight_scale(
    weight: nn.Parameter, num_shards: int, model_config: ModelConfig
):  # pylint: disable=unused-argument
    (spatial, red), dtype = weight.shape, weight.dtype
    spatial, red = int(spatial), int(red) * num_shards
    a = te.placeholder((spatial, red), dtype=dtype)
    w = topi.reshape(a, (spatial, num_shards, red // num_shards))
    w = topi.transpose(w, (1, 0, 2))
    func = te.create_prim_func([a, w])
    return func


def _shard_gate_up_weight_scale(
    weight: nn.Parameter, num_shards: int, model_config: ModelConfig
):  # pylint: disable=unused-argument
    (spatial, red), dtype = weight.shape, weight.dtype
    spatial, red = int(spatial) * num_shards, int(red)
    a = te.placeholder((spatial, red), dtype=dtype)
    g = te.compute((spatial // 2, red), lambda i, j: a[i, j])
    u = te.compute((spatial // 2, red), lambda i, j: a[spatial // 2 + i, j])
    g = topi.reshape(g, (num_shards, spatial // 2 // num_shards, red))
    u = topi.reshape(u, (num_shards, spatial // 2 // num_shards, red))
    w = topi.concatenate((g, u), axis=1)
    w = topi.reshape(w, (num_shards, spatial // num_shards, red))
    func = te.create_prim_func([a, w])
    return func
