import tvm
from tvm import relax, te, topi
from tvm.script import tir as T

from ..relax_model.llama import LlamaConfig

def emit_reorder_qvk_proj(bb: relax.BlockBuilder, config: LlamaConfig) -> None:
    def _emit(dtype: str, global_symbol: str, q_heads: int, kv_heads: int, head_dim: int):
        m = te.var("m", dtype="int64")
        n = te.var("n", dtype="int64")
        num_shards = te.var("num_shards", dtype="int64")

        weight = te.placeholder((m, n), dtype=dtype, name="proj_weight")
        q_weight = te.compute((q_heads * head_dim, n), lambda i, j: weight[i, j], name="split_q")
        k_weight = te.compute(
            (kv_heads * head_dim, n), lambda i, j: weight[q_heads * head_dim + i, j], name="split_k"
        )
        v_weight = te.compute(
            (kv_heads * head_dim, n),
            lambda i, j: weight[(q_heads + kv_heads) * head_dim + i, j],
            name="split_v",
        )
        q_heads_per_shard = q_heads // num_shards
        kv_heads_per_shard = kv_heads // num_shards
        q_weight = topi.reshape(q_weight, (num_shards, q_heads_per_shard, head_dim, n))
        k_weight = topi.reshape(k_weight, (num_shards, kv_heads_per_shard, head_dim, n))
        v_weight = topi.reshape(v_weight, (num_shards, kv_heads_per_shard, head_dim, n))

        concat_weight = topi.concatenate([q_weight, k_weight, v_weight], axis=1)
        result = topi.reshape(concat_weight, (m, n))

        reorder_qkv = te.create_prim_func(
            [weight, q_heads, kv_heads, head_dim, num_shards, result], index_dtype_override="int64"
        )

        bb.add_func(reorder_qkv, global_symbol)
    
    num_query_heads = config.num_attention_heads
    num_key_value_heads = (
        config.num_key_value_heads is None
        and config.num_attention_heads
        or config.num_key_value_heads
    )
    head_dim = config.hidden_size // config.num_attention_heads
    _emit("float32", "reorder_qkv_proj_fp32", num_query_heads, num_key_value_heads, head_dim)
    _emit("float16", "reorder_qkv_proj_fp16", num_query_heads, num_key_value_heads, head_dim)
    _emit("uint32", "reorder_qkv_proj_uint32", num_query_heads, num_key_value_heads, head_dim)


def emit_reorder_gate_up_proj(bb: relax.BlockBuilder, config: LlamaConfig) -> None:
    def _emit(dtype: str, global_symbol: str, intermediate_size: int):
        m = te.var("m", dtype="int64")
        n = te.var("n", dtype="int64")
        num_shards = te.var("num_shards", dtype="int64")

        weight = te.placeholder((m, n), dtype=dtype, name="proj_weight")
        gate_weight = te.compute((intermediate_size, n), lambda i, j: weight[i, j], name="split_gate")
        up_weight = te.compute(
            (intermediate_size, n), lambda i, j: weight[intermediate_size + i, j], name="split_up"
        )
        intermedia_size_per_shard = intermediate_size // num_shards
        gate_weight = topi.reshape(gate_weight, (num_shards, intermedia_size_per_shard, n))
        up_weight = topi.reshape(up_weight, (num_shards, intermedia_size_per_shard, n))
        concat_weight = topi.concatenate([gate_weight, up_weight], axis=1)
        result = topi.reshape(concat_weight, (m, n))

        reorder_gate_up_proj = te.create_prim_func([weight, num_shards, result], index_dtype_override="int64")

        bb.add_func(reorder_gate_up_proj, global_symbol)
    
    _emit("float32", "reorder_gate_up_proj_fp32", config.intermediate_size)
    _emit("float16", "reorder_gate_up_proj_fp16", config.intermediate_size)
    _emit("uint32", "reorder_gate_up_proj_uint32", config.intermediate_size)


def emit_shard3d(bb: relax.BlockBuilder) -> None:
    from tvm.script import tir as T

    def _emit(dtype: str, global_symbol: str):
        @T.prim_func
        def shard_3d(a: T.handle, num_shards: T.int64, b: T.handle):
            T.func_attr(
                {
                    "tir.noalias": T.bool(True),
                    "global_symbol": global_symbol,
                }
            )
            s_0, s_1, s_2 = T.int64(), T.int64(), T.int64()
            # pylint: disable=invalid-name
            A = T.match_buffer(a, (s_0, s_1, s_2), dtype)
            B = T.match_buffer(b, (num_shards, s_0, s_1 // num_shards, s_2), dtype)
            # pylint: enable=invalid-name
            for j_o, i, j_i, k in T.grid(num_shards, s_0, s_1 // num_shards, s_2):
                with T.block("B"):
                    v_j_o = T.axis.spatial(num_shards, j_o)
                    v_i = T.axis.spatial(s_0, i)
                    v_j_i = T.axis.spatial(s_1 // num_shards, j_i)
                    v_k = T.axis.spatial(s_2, k)
                    B[v_j_o, v_i, v_j_i, v_k] = A[v_i, v_j_o * (s_1 // num_shards) + v_j_i, v_k]

        bb.add_func(shard_3d, global_symbol)

    _emit("float32", "shard3d_fp32")
    _emit("float16", "shard3d_fp16")
    _emit("uint32", "shard3d_uint32")


def emit_shard_funcs(bb: relax.BlockBuilder, config: LlamaConfig):
    emit_shard3d(bb)
    emit_reorder_gate_up_proj(bb, config)
    emit_reorder_qvk_proj(bb, config)