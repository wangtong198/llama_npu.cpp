import math
import torch


def flash_attn_cpu_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = True,
    block_size: int = 128,
    scale: float | None = None,
    mask: torch.Tensor | None = None,
    logit_softcap: float | None = None,
):
    """
    输入:
      q: [batch_size, num_query_heads, num_query_tokens, head_dim]
      k: [num_kv_heads, num_key_tokens, head_dim] 或 [batch_size, num_kv_heads, num_key_tokens, head_dim]
      v: [num_kv_heads, num_key_tokens, head_dim] 或 [batch_size, num_kv_heads, num_key_tokens, head_dim]

    输出:
      [batch_size, num_query_heads, num_query_tokens, head_dim]

    约束:
      - 支持 GQA/MQA：num_query_heads 必须能整除 num_kv_heads
      - 使用 float32 做中间累加，优先保证数值稳定
      - mask 支持:
          * None
          * [num_query_tokens, num_key_tokens]
          * [batch_size, num_query_tokens, num_key_tokens]
        其中 bool mask 表示 True=保留，False=屏蔽；
        float mask 表示直接加到 logits 上，常见是 0 / -inf
    """
    assert q.dim() == 4, "q 必须是 [batch_size, num_query_heads, num_query_tokens, head_dim]"
    assert k.dim() in (3, 4), "k 必须是 [num_kv_heads, num_key_tokens, head_dim] 或 [batch_size, num_kv_heads, num_key_tokens, head_dim]"
    assert v.dim() in (3, 4), "v 必须是 [num_kv_heads, num_key_tokens, head_dim] 或 [batch_size, num_kv_heads, num_key_tokens, head_dim]"

    batch_size, num_query_heads, num_query_tokens, head_dim = q.shape

    if k.dim() == 3:
        k = k.unsqueeze(0).expand(batch_size, -1, -1, -1)
    if v.dim() == 3:
        v = v.unsqueeze(0).expand(batch_size, -1, -1, -1)

    assert k.shape[0] == batch_size and v.shape[0] == batch_size, "k/v 的 batch 维必须与 q 一致"
    assert k.shape[3] == head_dim and v.shape[3] == head_dim, "head_dim 必须一致"

    _, num_kv_heads, num_key_tokens, _ = k.shape
    assert num_query_heads % num_kv_heads == 0, "num_query_heads 必须能整除 num_kv_heads（GQA/MQA 映射）"
    query_heads_per_kv_head = num_query_heads // num_kv_heads

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    query_tensor = q.float()
    key_tensor = k.float()
    value_tensor = v.float()

    output = torch.empty((batch_size, num_query_heads, num_query_tokens, head_dim), device=q.device, dtype=torch.float32)

    query_positions = torch.arange(num_query_tokens, device=q.device)[:, None]   # [num_query_tokens, 1]
    key_positions = torch.arange(num_key_tokens, device=q.device)[None]         # [1, num_key_tokens]

    for batch_idx in range(batch_size):
        batch_mask = None
        if mask is not None:
            if mask.dim() == 2:
                batch_mask = mask
            elif mask.dim() == 3:
                batch_mask = mask[batch_idx]
            else:
                raise ValueError("mask 只能是 [num_query_tokens, num_key_tokens] 或 [batch_size, num_query_tokens, num_key_tokens]")

        for query_head_idx in range(num_query_heads):
            kv_head_idx = query_head_idx // query_heads_per_kv_head

            query_slice = query_tensor[batch_idx, query_head_idx]     # [num_query_tokens, head_dim]
            key_slice = key_tensor[batch_idx, kv_head_idx]            # [num_key_tokens, head_dim]
            value_slice = value_tensor[batch_idx, kv_head_idx]        # [num_key_tokens, head_dim]

            # 对应 CPU 实现中的行最大值、行归一化分母和输出累加器
            row_max = torch.full((num_query_tokens,), -float("inf"), device=q.device, dtype=torch.float32)
            row_sum = torch.zeros((num_query_tokens,), device=q.device, dtype=torch.float32)
            row_output = torch.zeros((num_query_tokens, head_dim), device=q.device, dtype=torch.float32)

            for block_start in range(0, num_key_tokens, block_size):
                block_end = min(block_start + block_size, num_key_tokens)

                key_block = key_slice[block_start:block_end]     # [block_size, head_dim]
                value_block = value_slice[block_start:block_end] # [block_size, head_dim]

                scores = query_slice @ key_block.T
                scores = scores * scale

                if logit_softcap is not None and logit_softcap != 0.0:
                    scores = logit_softcap * torch.tanh(scores / logit_softcap)

                if causal:
                    causal_block = key_positions[:, block_start:block_end] > query_positions
                    scores = scores.masked_fill(causal_block, -float("inf"))

                if batch_mask is not None:
                    mask_block = batch_mask[:, block_start:block_end]
                    if mask_block.dtype == torch.bool:
                        scores = scores.masked_fill(~mask_block, -float("inf"))
                    else:
                        scores = scores + mask_block

                block_max = scores.max(dim=-1).values
                valid_rows = torch.isfinite(block_max)

                # 某些行如果整块都被 mask 掉，max 会变成 -inf，需要单独处理
                normalized_scores = torch.where(
                    valid_rows[:, None],
                    scores - block_max[:, None],
                    torch.zeros_like(scores),
                )
                probabilities = torch.exp(normalized_scores) * valid_rows[:, None]
                block_sum = probabilities.sum(dim=-1)
                block_output = probabilities @ value_block

                new_row_max = torch.maximum(row_max, block_max)

                old_scale = torch.exp(row_max - new_row_max)
                block_scale = torch.exp(block_max - new_row_max)

                # 如果整行都被 mask 掉，新最大值仍然是 -inf，需要把缩放项归零
                invalid_rows = ~torch.isfinite(new_row_max)
                old_scale = torch.where(invalid_rows, torch.zeros_like(old_scale), old_scale)
                block_scale = torch.where(invalid_rows, torch.zeros_like(block_scale), block_scale)

                row_sum = old_scale * row_sum + block_scale * block_sum
                row_output = old_scale[:, None] * row_output + block_scale[:, None] * block_output
                row_max = new_row_max

            output[batch_idx, query_head_idx] = row_output / row_sum.clamp_min(1e-20)[:, None]

    return output.to(dtype=q.dtype)


def naive_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = True,
    scale: float | None = None,
    mask: torch.Tensor | None = None,
    logit_softcap: float | None = None,
):
    """
    用于对照验证的朴素 attention 实现。
    输入输出形状与 flash_attn_cpu_torch 保持一致。
    """
    assert q.dim() == 4, "q 必须是 [batch_size, num_query_heads, num_query_tokens, head_dim]"
    batch_size, num_query_heads, num_query_tokens, head_dim = q.shape

    if k.dim() == 3:
        k = k.unsqueeze(0).expand(batch_size, -1, -1, -1)
    if v.dim() == 3:
        v = v.unsqueeze(0).expand(batch_size, -1, -1, -1)

    _, num_kv_heads, num_key_tokens, _ = k.shape
    assert num_query_heads % num_kv_heads == 0, "num_query_heads 必须能整除 num_kv_heads（GQA/MQA 映射）"
    query_heads_per_kv_head = num_query_heads // num_kv_heads

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    query_tensor = q.float()
    key_tensor = k.float()
    value_tensor = v.float()

    output = torch.empty((batch_size, num_query_heads, num_query_tokens, head_dim), device=q.device, dtype=torch.float32)

    for batch_idx in range(batch_size):
        batch_mask = None
        if mask is not None:
            batch_mask = mask if mask.dim() == 2 else mask[batch_idx]

        for query_head_idx in range(num_query_heads):
            kv_head_idx = query_head_idx // query_heads_per_kv_head
            scores = query_tensor[batch_idx, query_head_idx] @ key_tensor[batch_idx, kv_head_idx].T
            scores = scores * scale

            if logit_softcap is not None and logit_softcap != 0.0:
                scores = logit_softcap * torch.tanh(scores / logit_softcap)

            if causal:
                causal_mask = torch.triu(
                    torch.ones((num_query_tokens, num_key_tokens), device=q.device, dtype=torch.bool),
                    diagonal=1,
                )
                scores = scores.masked_fill(causal_mask, -float("inf"))

            if batch_mask is not None:
                if batch_mask.dtype == torch.bool:
                    scores = scores.masked_fill(~batch_mask, -float("inf"))
                else:
                    scores = scores + batch_mask

            probabilities = torch.softmax(scores, dim=-1)
            output[batch_idx, query_head_idx] = probabilities @ value_tensor[batch_idx, kv_head_idx]

    return output.to(dtype=q.dtype)


if __name__ == "__main__":
    torch.manual_seed(0)

    device = "cpu"
    dtype = torch.float32

    batch_size = 2
    num_query_heads = 4
    num_kv_heads = 2
    num_query_tokens = 7
    num_key_tokens = 9
    head_dim = 8

    query = torch.randn(batch_size, num_query_heads, num_query_tokens, head_dim, device=device, dtype=dtype)
    key = torch.randn(num_kv_heads, num_key_tokens, head_dim, device=device, dtype=dtype)
    value = torch.randn(num_kv_heads, num_key_tokens, head_dim, device=device, dtype=dtype)

    # causal 场景下不额外传 mask，直接依赖 causal=True
    mask = None

    y_flash = flash_attn_cpu_torch(
        query, key, value,
        causal=True,
        block_size=3,
        mask=mask,
        logit_softcap=None,
    )

    y_ref = naive_attention(
        query, key, value,
        causal=True,
        mask=mask,
        logit_softcap=None,
    )

    max_abs_err = (y_flash - y_ref).abs().max().item()
    print("max_abs_err =", max_abs_err)

    assert torch.allclose(y_flash, y_ref, atol=1e-5, rtol=1e-5), "结果不一致"
    print("OK")