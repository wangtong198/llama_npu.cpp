from __future__ import annotations

import unittest

import torch


def _validate_mul_mat_id_inputs(
    x: torch.Tensor,
    weight: torch.Tensor,
    ids: torch.Tensor,
) -> tuple[int, int, int, int, int]:
    """
    Validate shapes for a torch version of ggml_mul_mat_id.

    Torch layout used in this file:
      x      : [n_tokens, input_slots, hidden_size]
      weight : [n_experts, intermediate_size, hidden_size]
      ids    : [n_tokens, top_k]

    Mapping to ggml_mul_mat_id:
      as  -> [cols, rows, n_expert]        <=> weight.transpose layout
      b   -> [cols, input_slots, n_tokens] <=> x.transpose layout
      ids -> [top_k, n_tokens]             <=> ids.transpose layout
      out -> [rows, top_k, n_tokens]       <=> output.transpose layout
    """
    if x.dim() != 3:
        raise ValueError(
            f"x must be 3D [n_tokens, input_slots, hidden_size], got shape {tuple(x.shape)}"
        )
    if weight.dim() != 3:
        raise ValueError(
            f"weight must be 3D [n_experts, intermediate_size, hidden_size], got shape {tuple(weight.shape)}"
        )
    if ids.dim() != 2:
        raise ValueError(
            f"ids must be 2D [n_tokens, top_k], got shape {tuple(ids.shape)}"
        )

    n_tokens, input_slots, hidden_size = x.shape
    n_experts, intermediate_size, weight_hidden_size = weight.shape
    ids_tokens, top_k = ids.shape

    if ids_tokens != n_tokens:
        raise ValueError(
            f"ids and x must have the same n_tokens, got {ids_tokens} vs {n_tokens}"
        )
    if weight_hidden_size != hidden_size:
        raise ValueError(
            "weight hidden_size must match x hidden_size, "
            f"got {weight_hidden_size} vs {hidden_size}"
        )
    if input_slots <= 0:
        raise ValueError("input_slots must be > 0")
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    if top_k % input_slots != 0:
        raise ValueError(
            "top_k must be divisible by input_slots to match ggml broadcast semantics, "
            f"got top_k={top_k}, input_slots={input_slots}"
        )

    if ids.numel() > 0:
        ids_min = int(ids.min().item())
        ids_max = int(ids.max().item())
        if ids_min < 0 or ids_max >= n_experts:
            raise ValueError(
                f"ids must be in [0, {n_experts - 1}], got min={ids_min}, max={ids_max}"
            )

    return n_tokens, input_slots, hidden_size, n_experts, intermediate_size


def mul_mat_id_torch(
    x: torch.Tensor,
    weight: torch.Tensor,
    ids: torch.Tensor,
) -> torch.Tensor:
    """
    Torch implementation of the ggml CPU GGML_OP_MUL_MAT_ID semantics.

    Input:
      x      : [n_tokens, input_slots, hidden_size]
      weight : [n_experts, intermediate_size, hidden_size]
      ids    : [n_tokens, top_k]

    Output:
      out    : [n_tokens, top_k, intermediate_size]

    Semantics:
      For every token t and top-k slot k:
        expert_id = ids[t, k]
        input_slot = k % input_slots
        out[t, k] = weight[expert_id] @ x[t, input_slot]

    This function intentionally mirrors the CPU implementation's execution idea:
      1. Scan ids and group requests by expert id.
      2. For each expert, gather all (slot, token) pairs assigned to it.
      3. Run the same expert matrix over all gathered inputs.
      4. Scatter the results back to [n_tokens, top_k, intermediate_size].
    """
    n_tokens, input_slots, hidden_size, n_experts, intermediate_size = _validate_mul_mat_id_inputs(
        x, weight, ids
    )

    compute_dtype = torch.promote_types(x.dtype, weight.dtype)
    x_compute = x.to(dtype=compute_dtype)
    weight_compute = weight.to(dtype=compute_dtype)

    top_k = ids.shape[1]
    out = torch.empty(
        (n_tokens, top_k, intermediate_size),
        device=x.device,
        dtype=compute_dtype,
    )

    expert_to_requests: list[list[tuple[int, int]]] = [[] for _ in range(n_experts)]

    for token_idx in range(n_tokens):
        for topk_slot in range(top_k):
            # 只对指定 token 的 top_k 个 expert 进行计算，其他 expert 不参与计算。
            expert_idx = int(ids[token_idx, topk_slot].item())
            expert_to_requests[expert_idx].append((topk_slot, token_idx))

    for expert_idx, requests in enumerate(expert_to_requests):
        if not requests:    # 未使用的 experts 不参与计算。
            continue

        slot_indices = torch.tensor(    # 取出 当前expert 关联的所有 topk_slot 索引
            [slot for slot, _ in requests],
            device=ids.device,
            dtype=torch.long,
        )
        token_indices = torch.tensor(    # 取出 当前expert 关联的所有 token
            [token for _, token in requests],
            device=ids.device,
            dtype=torch.long,
        )

        # 要求可广播到 top_k 个 expert 上。
        # 例如：input_slots = 1，slot_indices = [1, 3, 4]，则 input_slot_indices = [0, 0, 0]
        # 等价于：
        # expert_inputs = torch.stack([
        #     x_compute[1, 0],
        #     x_compute[3, 0],
        #     x_compute[4, 0],
        # ], dim=0)
        input_slot_indices = slot_indices % input_slots

        expert_inputs = x_compute[token_indices, input_slot_indices] # 
        expert_weight = weight_compute[expert_idx]

        # [num_requests, hidden_size] @ [hidden_size, intermediate_size]
        expert_outputs = expert_inputs @ expert_weight.transpose(0, 1)  # [num_requests, intermediate_size]

        # out shape:[n_tokens, top_k, intermediate_size]
        # 将第 expert_idx 个 expert 的计算结果，写入到 out 的第 token_indices, slot_indices 位置。
        # 涉及 PyTorch 的高级索引赋值
        # 按 (token_idx, slot_idx) 这一对一对的位置，把 expert_outputs 里的每一行写回 out 的对应位置。
        out[token_indices, slot_indices] = expert_outputs

    return out


def mul_mat_id_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    ids: torch.Tensor,
) -> torch.Tensor:
    """
    Slow reference implementation used by tests.
    """
    n_tokens, input_slots, _, _, intermediate_size = _validate_mul_mat_id_inputs(
        x, weight, ids
    )

    compute_dtype = torch.promote_types(x.dtype, weight.dtype)
    x_compute = x.to(dtype=compute_dtype)
    weight_compute = weight.to(dtype=compute_dtype)

    top_k = ids.shape[1]
    out = torch.empty(
        (n_tokens, top_k, intermediate_size),
        device=x.device,
        dtype=compute_dtype,
    )

    for token_idx in range(n_tokens):
        for topk_slot in range(top_k):
            expert_idx = int(ids[token_idx, topk_slot].item())
            input_slot = topk_slot % input_slots
            out[token_idx, topk_slot] = (
                weight_compute[expert_idx] @ x_compute[token_idx, input_slot]
            )

    return out


class MulMatIdTorchTest(unittest.TestCase):
    def test_small_handcrafted_case(self) -> None:
        x = torch.tensor(
            [
                [[1.0, 2.0]],
                [[3.0, 4.0]],
            ]
        )
        weight = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[2.0, 0.0], [0.0, 3.0]],
                [[1.0, 1.0], [1.0, -1.0]],
            ]
        )
        ids = torch.tensor(
            [
                [0, 2],
                [1, 0],
            ],
            dtype=torch.int64,
        )

        actual = mul_mat_id_torch(x, weight, ids)
        expected = torch.tensor(
            [
                [[1.0, 2.0], [3.0, -1.0]],
                [[6.0, 12.0], [3.0, 4.0]],
            ]
        )

        self.assertTrue(torch.allclose(actual, expected))

    def test_matches_reference_single_input_slot(self) -> None:
        torch.manual_seed(0)

        n_tokens = 5
        hidden_size = 7
        intermediate_size = 11
        n_experts = 4
        top_k = 3

        x = torch.randn(n_tokens, 1, hidden_size)
        weight = torch.randn(n_experts, intermediate_size, hidden_size)
        ids = torch.randint(0, n_experts, (n_tokens, top_k), dtype=torch.int64)

        actual = mul_mat_id_torch(x, weight, ids)
        expected = mul_mat_id_reference(x, weight, ids)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))

    def test_matches_reference_with_broadcastable_input_slots(self) -> None:
        torch.manual_seed(1)

        n_tokens = 4
        input_slots = 2
        hidden_size = 6
        intermediate_size = 9
        n_experts = 5
        top_k = 4

        x = torch.randn(n_tokens, input_slots, hidden_size)
        weight = torch.randn(n_experts, intermediate_size, hidden_size)
        ids = torch.randint(0, n_experts, (n_tokens, top_k), dtype=torch.int32)

        actual = mul_mat_id_torch(x, weight, ids)
        expected = mul_mat_id_reference(x, weight, ids)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))

    def test_repeated_expert_ids(self) -> None:
        torch.manual_seed(2)

        x = torch.randn(3, 1, 8)
        weight = torch.randn(4, 10, 8)
        ids = torch.tensor(
            [
                [2, 2, 2],
                [1, 1, 3],
                [0, 0, 0],
            ],
            dtype=torch.int64,
        )

        actual = mul_mat_id_torch(x, weight, ids)
        expected = mul_mat_id_reference(x, weight, ids)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))

    def test_invalid_top_k_broadcast(self) -> None:
        x = torch.randn(2, 2, 4)
        weight = torch.randn(3, 5, 4)
        ids = torch.randint(0, 3, (2, 3), dtype=torch.int64)

        with self.assertRaisesRegex(ValueError, "top_k must be divisible by input_slots"):
            mul_mat_id_torch(x, weight, ids)

    def test_invalid_expert_id_range(self) -> None:
        x = torch.randn(2, 1, 4)
        weight = torch.randn(3, 5, 4)
        ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)

        with self.assertRaisesRegex(ValueError, "ids must be in"):
            mul_mat_id_torch(x, weight, ids)


if __name__ == "__main__":
    unittest.main()
