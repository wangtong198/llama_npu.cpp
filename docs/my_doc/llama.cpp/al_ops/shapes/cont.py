import math
import torch

def scaled_dot_product_attention(
    query, key, value, cont = False, attn_mask=None, dropout_p=0.0,
    is_causal=False, scale=None, enable_gqa=False,
) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias
    if enable_gqa:
        key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)
    
    if cont:
        Kt = key.transpose(-2, -1).contiguous()
        print(f"Kt.shape: {Kt.shape}, Kt.stride(): {Kt.stride()}")
    else:
        Kt = key.transpose(-2, -1)
        print(f"Kt.shape: {Kt.shape}, Kt.stride(): {Kt.stride()}")
    attn_weight = query @ Kt * scale_factor
    print(f"attn_weight.shape: {attn_weight.shape}, attn_weight.stride(): {attn_weight.stride()}")
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


def run_contiguous_matmul_experiment():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    B, H, L, S, D = 2, 4, 16, 32, 64
    Q = torch.randn(B, H, L, D, device=device, dtype=dtype)
    K = torch.randn(B, H, S, D, device=device, dtype=dtype)
    V = torch.randn(B, H, S, D, device=device, dtype=dtype)
    # Kt = K.transpose(-2, -1)  # (B, H, D, S)
    # Kt_c = Kt.contiguous()
    # print("Kt.is_contiguous():", Kt.is_contiguous())
    # print("Kt_c.is_contiguous():", Kt_c.is_contiguous())
    # print("Kt.stride():", Kt.stride(), "| Kt_c.stride():", Kt_c.stride())
    scale = 1 / math.sqrt(D)
    # scores_nc = Q @ Kt * scale
    # scores_c = Q @ Kt_c * scale
    # print("scores_nc.is_contiguous():", scores_nc.is_contiguous())
    # print("scores_c.is_contiguous():", scores_c.is_contiguous())
    # print("max |scores_nc - scores_c|:", (scores_nc - scores_c).abs().max().item())
    # print("allclose:", torch.allclose(scores_nc, scores_c))

    print(f"=====================================")
    # 走完整 SDPA（关 dropout，避免随机）
    out_nc = scaled_dot_product_attention(
        Q, K, V, cont = False, attn_mask=None, dropout_p=0.0, is_causal=False, scale=scale, enable_gqa=False
    )
    # 强制 K 侧 contiguous：在函数外先 contiguous 再传参做不到“只改 Kᵀ”，
    # 所以这里用同一套 Q,K,V，另写一行对比：手工替换 attn 里那一步等价于 K contiguous 的 Kᵀ
    # Kt2 = K.transpose(-2, -1).contiguous()
    # attn_weight2 = torch.softmax(Q @ Kt2 * scale, dim=-1)
    # out_c_manual = attn_weight2 @ V
    # print("out_nc vs out_c_manual max diff:", (out_nc - out_c_manual).abs().max().item())
    # print("out_nc.is_contiguous():", out_nc.is_contiguous())

    print(f"=====================================")
    out_nc1 = scaled_dot_product_attention(
        Q, K, V, cont = True, attn_mask=None, dropout_p=0.0, is_causal=False, scale=scale, enable_gqa=False
    )


    # 强制 K 侧 contiguous：在函数外先 contiguous 再传参做不到“只改 Kᵀ”，
    # 所以这里用同一套 Q,K,V，另写一行对比：手工替换 attn 里那一步等价于 K contiguous 的 Kᵀ
    # Kt2 = K.transpose(-2, -1).contiguous()
    # attn_weight2 = torch.softmax(Q @ Kt2 * scale, dim=-1)
    # out_c_manual = attn_weight2 @ V
    # print("out_nc1 vs out_c_manual max diff:", (out_nc1 - out_c_manual).abs().max().item())
    # print("out_nc1.is_contiguous():", out_nc1.is_contiguous())
if __name__ == "__main__":
    # run_contiguous_matmul_experiment()

    A = torch.randn(2, 3)
    B = torch.randn(3, 4)

    C = A@B
    print(f"C.shape: {C.shape}, C.stride(): {C.stride()}")
    
    # 底层一块 (2, 8) 的连续存储，只取偶数列 → 形状 (2, 4)，最后一维步长为 2，非连续
    base = torch.randn(4, 2)
    print(base.stride()) 
    out = base.transpose(0, 1)

    print(out.stride()) 

    assert out.shape == (2, 4)
    assert not out.is_contiguous()  # stride 类似 (8, 2)，末维不连续

    torch.matmul(A, B, out=out)

    assert not out.is_contiguous()  # 乘完仍然继承 out 的布局
    print(out.stride())  # 例如 (8, 2)

    out = out.contiguous()
    assert out.is_contiguous() 
    print(out.stride()) 