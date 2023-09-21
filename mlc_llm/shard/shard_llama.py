import tvm
from tvm import relax
from tvm.script import tir as T

def emit_reorder_qvk_proj(bb: relax.BlockBuilder) -> None:
    def _emit(dtype: str, global_symbol: str):
        @T.prim_func
        def reorder_qkv_proj(var_proj_weight: T.handle, q_end: T.int64, k_end: T.int64, head_dim: T.int64, num_shards: T.int64, var_output: T.handle):
            T.func_attr({"tir.noalias": T.bool(True), "global_symbol": global_symbol})
            m, n = T.int64(), T.int64()
            proj_weight = T.match_buffer(var_proj_weight, (m, n), dtype)
            output = T.match_buffer(var_output, (m, n), dtype)
            # with T.block("root"):
            split_q = T.alloc_buffer((q_end, n))
            reshape_q = T.alloc_buffer((num_shards, q_end // head_dim // num_shards, head_dim, n))
            split_v = T.alloc_buffer((k_end - q_end, n))
            reshape_k = T.alloc_buffer((num_shards, (k_end - q_end) // head_dim // num_shards, head_dim, n))
            split_v_1 = T.alloc_buffer((m - k_end, n))
            reshape_v = T.alloc_buffer((num_shards, (m - k_end) // head_dim // num_shards, head_dim, n))
            concat_qkv = T.alloc_buffer((num_shards, q_end // head_dim // num_shards + (k_end - q_end) // head_dim // num_shards + (m - k_end) // head_dim // num_shards, head_dim, n))
            for i, j in T.grid(q_end, n):
                with T.block("split_q"):
                    v_i = T.axis.spatial(q_end, i)
                    v_j = T.axis.spatial(n, j)
                    T.reads(proj_weight[v_i, v_j])
                    T.writes(split_q[v_i, v_j])
                    split_q[v_i, v_j] = proj_weight[v_i, v_j]
            for i, j, k, l in T.grid(num_shards, q_end // head_dim // num_shards, head_dim, n):
                with T.block("reshape_q"):
                    v_i = T.axis.spatial(num_shards, i)
                    v_j = T.axis.spatial(q_end // head_dim // num_shards, j)
                    v_k = T.axis.spatial(head_dim, k)
                    v_l = T.axis.spatial(n, l)
                    T.reads(split_q[(v_i * (q_end // head_dim // num_shards) + v_j) * head_dim + v_k, v_l])
                    T.writes(reshape_q[v_i, v_j, v_k, v_l])
                    reshape_q[v_i, v_j, v_k, v_l] = split_q[(v_i * (q_end // head_dim // num_shards) + v_j) * head_dim + v_k, v_l]
            for i, j in T.grid(k_end - q_end, n):
                with T.block("split_v"):
                    v_i = T.axis.spatial(k_end - q_end, i)
                    v_j = T.axis.spatial(n, j)
                    T.reads(proj_weight[q_end + v_i, v_j])
                    T.writes(split_v[v_i, v_j])
                    split_v[v_i, v_j] = proj_weight[q_end + v_i, v_j]
            for i, j, k, l in T.grid(num_shards, (k_end - q_end) // head_dim // num_shards, head_dim, n):
                with T.block("reshape_k"):
                    v_i = T.axis.spatial(num_shards, i)
                    v_j = T.axis.spatial((k_end - q_end) // head_dim // num_shards, j)
                    v_k = T.axis.spatial(head_dim, k)
                    v_l = T.axis.spatial(n, l)
                    T.reads(split_v[(v_i * ((k_end - q_end) // head_dim // num_shards) + v_j) * head_dim + v_k, v_l])
                    T.writes(reshape_k[v_i, v_j, v_k, v_l])
                    reshape_k[v_i, v_j, v_k, v_l] = split_v[(v_i * ((k_end - q_end) // head_dim // num_shards) + v_j) * head_dim + v_k, v_l]
            for i, j in T.grid(m - k_end, n):
                with T.block("split_v_1"):
                    v_i = T.axis.spatial(m - k_end, i)
                    v_j = T.axis.spatial(n, j)
                    T.reads(proj_weight[k_end + v_i, v_j])
                    T.writes(split_v_1[v_i, v_j])
                    split_v_1[v_i, v_j] = proj_weight[k_end + v_i, v_j]
            for i, j, k, l in T.grid(num_shards, (m - k_end) // head_dim // num_shards, head_dim, n):
                with T.block("reshape_v"):
                    v_i = T.axis.spatial(num_shards, i)
                    v_j = T.axis.spatial((m - k_end) // head_dim // num_shards, j)
                    v_k = T.axis.spatial(head_dim, k)
                    v_l = T.axis.spatial(n, l)
                    T.reads(split_v_1[(v_i * ((m - k_end) // head_dim // num_shards) + v_j) * head_dim + v_k, v_l])
                    T.writes(reshape_v[v_i, v_j, v_k, v_l])
                    reshape_v[v_i, v_j, v_k, v_l] = split_v_1[(v_i * ((m - k_end) // head_dim // num_shards) + v_j) * head_dim + v_k, v_l]
            for i, j, k, l in T.grid(num_shards, q_end // head_dim // num_shards + (k_end - q_end) // head_dim // num_shards + (m - k_end) // head_dim // num_shards, head_dim, n):
                with T.block("concat_qkv"):
                    v_i = T.axis.spatial(num_shards, i)
                    v_j = T.axis.spatial(q_end // head_dim // num_shards + (k_end - q_end) // head_dim // num_shards + (m - k_end) // head_dim // num_shards, j)
                    v_k = T.axis.spatial(head_dim, k)
                    v_l = T.axis.spatial(n, l)
                    T.reads(reshape_q[v_i, v_j, v_k, v_l], reshape_k[v_i, v_j - q_end // head_dim // num_shards, v_k, v_l], reshape_v[v_i, v_j - q_end // head_dim // num_shards - (k_end - q_end) // head_dim // num_shards, v_k, v_l])
                    T.writes(concat_qkv[v_i, v_j, v_k, v_l])
                    concat_qkv[v_i, v_j, v_k, v_l] = T.if_then_else(v_j < q_end // head_dim // num_shards, reshape_q[v_i, v_j, v_k, v_l], T.if_then_else(v_j < q_end // head_dim // num_shards + (k_end - q_end) // head_dim // num_shards, reshape_k[v_i, v_j - q_end // head_dim // num_shards, v_k, v_l], reshape_v[v_i, v_j - q_end // head_dim // num_shards - (k_end - q_end) // head_dim // num_shards, v_k, v_l]))
            for i, j in T.grid(m, n):
                with T.block("output"):
                    v_i = T.axis.spatial(m, i)
                    v_j = T.axis.spatial(n, j)
                    T.reads(concat_qkv[v_i // (m // num_shards), v_i % (m // num_shards) // head_dim, v_i % (m // num_shards) % head_dim, v_j])
                    T.writes(output[v_i, v_j])
                    output[v_i, v_j] = concat_qkv[v_i // (m // num_shards), v_i % (m // num_shards) // head_dim, v_i % (m // num_shards) % head_dim, v_j]
        
        bb.add_func(reorder_qkv_proj, global_symbol)
    
    _emit("float32", "reorder_qkv_proj_fp32")
    _emit("float16", "reorder_qkv_proj_fp16")
    _emit("uint32", "reorder_qkv_proj_uint32")


def emit_reorder_gate_up_proj(bb: relax.BlockBuilder) -> None:
    def _emit(dtype: str, global_symbol: str):
        @T.prim_func
        def reorder_gate_up_proj(var_proj_weight: T.handle, gate_end: T.int64, num_shards: T.int64, var_output: T.handle):
            T.func_attr({"tir.noalias": T.bool(True), "global_symbol": global_symbol})
            m, n = T.int64(), T.int64()
            proj_weight = T.match_buffer(var_proj_weight, (m, n), dtype)
            output = T.match_buffer(var_output, (m, n), dtype)
            # with T.block("root"):
            split_gate = T.alloc_buffer((gate_end, n))
            reshape_gate = T.alloc_buffer((num_shards, gate_end // num_shards, n))
            split_up = T.alloc_buffer((gate_end, n))
            reshape_up = T.alloc_buffer((num_shards, gate_end // num_shards, n))
            concat_gate_up = T.alloc_buffer((num_shards, gate_end // num_shards + gate_end // num_shards, n))
            for i, j in T.grid(gate_end, n):
                with T.block("split_gate"):
                    v_i = T.axis.spatial(gate_end, i)
                    v_j = T.axis.spatial(n, j)
                    T.reads(proj_weight[v_i, v_j])
                    T.writes(split_gate[v_i, v_j])
                    split_gate[v_i, v_j] = proj_weight[v_i, v_j]
            for i, j, k in T.grid(num_shards, gate_end // num_shards, n):
                with T.block("reshape_gate"):
                    v_i = T.axis.spatial(num_shards, i)
                    v_j = T.axis.spatial(gate_end // num_shards, j)
                    v_k = T.axis.spatial(n, k)
                    T.reads(split_gate[v_i * (gate_end // num_shards) + v_j, v_k])
                    T.writes(reshape_gate[v_i, v_j, v_k])
                    reshape_gate[v_i, v_j, v_k] = split_gate[v_i * (gate_end // num_shards) + v_j, v_k]
            for i, j in T.grid(gate_end, n):
                with T.block("split_up"):
                    v_i = T.axis.spatial(gate_end, i)
                    v_j = T.axis.spatial(n, j)
                    T.reads(proj_weight[gate_end + v_i, v_j])
                    T.writes(split_up[v_i, v_j])
                    split_up[v_i, v_j] = proj_weight[gate_end + v_i, v_j]
            for i, j, k in T.grid(num_shards, gate_end // num_shards, n):
                with T.block("reshape_up"):
                    v_i = T.axis.spatial(num_shards, i)
                    v_j = T.axis.spatial(gate_end // num_shards, j)
                    v_k = T.axis.spatial(n, k)
                    T.reads(split_up[v_i * (gate_end // num_shards) + v_j, v_k])
                    T.writes(reshape_up[v_i, v_j, v_k])
                    reshape_up[v_i, v_j, v_k] = split_up[v_i * (gate_end // num_shards) + v_j, v_k]
            for i, j, k in T.grid(num_shards, gate_end // num_shards * T.int64(2), n):
                with T.block("concat_gate_up"):
                    v_i = T.axis.spatial(num_shards, i)
                    v_j = T.axis.spatial(gate_end // num_shards * T.int64(2), j)
                    v_k = T.axis.spatial(n, k)
                    T.reads(reshape_gate[v_i, v_j, v_k], reshape_up[v_i, v_j - gate_end // num_shards, v_k])
                    T.writes(concat_gate_up[v_i, v_j, v_k])
                    concat_gate_up[v_i, v_j, v_k] = T.if_then_else(v_j < gate_end // num_shards, reshape_gate[v_i, v_j, v_k], reshape_up[v_i, v_j - gate_end // num_shards, v_k])
            for i, j in T.grid(m, n):
                with T.block("output"):
                    v_i = T.axis.spatial(m, i)
                    v_j = T.axis.spatial(n, j)
                    T.reads(concat_gate_up[v_i // (gate_end // num_shards * T.int64(2)), v_i % (gate_end // num_shards * T.int64(2)), v_j])
                    T.writes(output[v_i, v_j])
                    output[v_i, v_j] = concat_gate_up[v_i // (gate_end // num_shards * T.int64(2)), v_i % (gate_end // num_shards * T.int64(2)), v_j]

        bb.add_func(reorder_gate_up_proj, global_symbol)
    
    _emit("float32", "reorder_gate_up_proj_fp32")
    _emit("float16", "reorder_gate_up_proj_fp16")
    _emit("uint32", "reorder_gate_up_proj_uint32")


def emit_reorder_llama_params(bb: relax.BlockBuilder):
    emit_reorder_gate_up_proj(bb)
    emit_reorder_qvk_proj(bb)