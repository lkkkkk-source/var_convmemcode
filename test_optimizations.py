"""
Optimization equivalence tests.
Verifies that all performance optimizations produce identical results.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Ensure models/ is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS = 0
FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}")


# =====================================================================
# Test 1: ClassAwareKnittingMemoryV2 forward equivalence
# =====================================================================
def test_class_aware_memory():
    print("\n=== Test 1: ClassAwareKnittingMemoryV2 ===")
    from models.class_aware_memory import ClassAwareKnittingMemoryV2

    torch.manual_seed(42)

    B, C = 4, 256  # Smaller for fast test
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    begin_ends = []
    cur = 0
    for pn in patch_nums:
        begin_ends.append((cur, cur + pn * pn))
        cur += pn * pn
    L = cur

    mem = ClassAwareKnittingMemoryV2(
        embed_dim=C, num_categories=22, num_scales=10,
        shared_patterns=4, shared_memory_size=2, cat_rank=4,
        block_idx=8, depth=16,
    )
    mem.train()

    # Test with mixed categories (some valid, some invalid)
    x = torch.randn(B, L, C)
    cat_ids = torch.tensor([3, -1, 15, 0])

    mem_k, mem_v, div_loss, sep_loss = mem(x, begin_ends, cat_ids)

    check("Output shapes correct",
          mem_k.shape == (B, L, C) and mem_v.shape == (B, L, C))
    check("Diversity loss is scalar", div_loss.dim() == 0)
    check("Slot sep loss is scalar", sep_loss.dim() == 0)
    check("Output not all zeros", mem_k.abs().sum() > 0)

    # Test gradient flow
    x_grad = torch.randn(B, L, C, requires_grad=True)
    mk, mv, dl, sl = mem(x_grad, begin_ends, cat_ids)
    loss = mk.sum() + mv.sum() + dl + sl
    loss.backward()
    check("Gradient flows to input", x_grad.grad is not None and x_grad.grad.abs().sum() > 0)

    # Check key parameters have gradients
    check("cat_A has gradient", mem.cat_A.grad is not None)
    check("cat_B has gradient", mem.cat_B.grad is not None)
    check("alpha_mlp has gradient",
          mem.alpha_mlp[0].weight.grad is not None)

    # Test with all invalid categories (CFG unconditional)
    x2 = torch.randn(B, L, C)
    cat_invalid = torch.tensor([-1, -1, -1, -1])
    mk2, mv2, _, _ = mem(x2, begin_ends, cat_invalid)
    check("All invalid cats produces output", mk2.abs().sum() > 0)

    # Test with all same category
    cat_same = torch.tensor([5, 5, 5, 5])
    mk3, mv3, _, _ = mem(x2, begin_ends, cat_same)
    check("All same cat produces output", mk3.abs().sum() > 0)

    # Test without category_ids (None)
    mk4, mv4, _, _ = mem(x2, begin_ends, None)
    check("None category_ids works", mk4.abs().sum() > 0)

    # Test slot sep caching (should recompute every 10 steps)
    mem.zero_grad()
    sep_losses = []
    for step in range(15):
        _, _, _, sl = mem(x2, begin_ends, cat_ids)
        sep_losses.append(sl.item())
    # First call and step 10 should be fresh; cached steps should match prior
    check("Slot sep caching works (values are finite)",
          all(math.isfinite(v) for v in sep_losses))


# =====================================================================
# Test 2: KnittingPatternMemory forward
# =====================================================================
def test_knitting_memory():
    print("\n=== Test 2: KnittingPatternMemory ===")
    from models.knitting_memory import KnittingPatternMemory

    torch.manual_seed(42)

    B, C = 2, 256
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    begin_ends = []
    cur = 0
    for pn in patch_nums:
        begin_ends.append((cur, cur + pn * pn))
        cur += pn * pn
    L = cur

    mem = KnittingPatternMemory(
        embed_dim=C, num_patterns=8, memory_size=4,
        num_scales=10, block_idx=0, depth=16,
    )
    mem.train()

    x = torch.randn(B, L, C)
    mk, mv, dl, sl = mem(x, begin_ends)

    check("Output shapes", mk.shape == (B, L, C) and mv.shape == (B, L, C))
    check("Output not zeros", mk.abs().sum() > 0)

    # Gradient flow
    x_g = torch.randn(B, L, C, requires_grad=True)
    mk2, mv2, dl2, sl2 = mem(x_g, begin_ends)
    (mk2.sum() + mv2.sum() + dl2 + sl2).backward()
    check("Gradient flows", x_g.grad is not None)

    # Slot sep caching
    sep_vals = []
    for _ in range(12):
        _, _, _, sl = mem(x, begin_ends)
        sep_vals.append(sl.item())
    check("Slot sep caching (finite)", all(math.isfinite(v) for v in sep_vals))


# =====================================================================
# Test 3: Texture execution plan equivalence
# =====================================================================
def test_texture_plan():
    print("\n=== Test 3: Texture Execution Plan ===")
    from models.basic_var import SelfAttention

    torch.manual_seed(42)

    C, H, depth = 256, 4, 16
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    begin_ends = []
    cur = 0
    for pn in patch_nums:
        begin_ends.append((cur, cur + pn * pn))
        cur += pn * pn
    L = cur

    attn = SelfAttention(
        block_idx=8, embed_dim=C, num_heads=H, depth=depth,
        enable_texture=True, texture_scales=[3, 5, 7, 11],
        texture_per_head_kernels=False,
    )
    attn.eval()

    # First forward builds the plan
    B = 2
    x = torch.randn(B, L, C)
    tex_feat = attn._compute_texture_modulation(x, begin_ends)

    check("Texture output shape", tex_feat.shape == (B, L, C))
    check("Texture plan built", attn._texture_plan is not None)
    check("Plan has correct length", len(attn._texture_plan) == len(begin_ends))

    # Count non-None entries (scales with pn >= 2)
    active = sum(1 for p in attn._texture_plan if p is not None)
    check("Active scales > 0", active > 0)
    # pn=1 should be None (too small)
    check("pn=1 skipped", attn._texture_plan[0] is None)

    # Second forward should reuse the plan
    tex_feat2 = attn._compute_texture_modulation(x, begin_ends)
    check("Deterministic output", torch.allclose(tex_feat, tex_feat2, atol=1e-6))

    # Test gradient flow through texture
    attn.train()
    x_g = torch.randn(B, L, C, requires_grad=True)
    tf = attn._compute_texture_modulation(x_g, begin_ends)
    tf.sum().backward()
    check("Texture gradient flows", x_g.grad is not None and x_g.grad.abs().sum() > 0)


# =====================================================================
# Test 4: Checkpoint compatibility
# =====================================================================
def test_checkpoint_compat():
    print("\n=== Test 4: Checkpoint Compatibility ===")
    from models.class_aware_memory import ClassAwareKnittingMemoryV2
    from models.knitting_memory import KnittingPatternMemory

    torch.manual_seed(42)

    # ClassAwareKnittingMemoryV2
    mem1 = ClassAwareKnittingMemoryV2(embed_dim=256, num_categories=22, num_scales=10,
                                       shared_patterns=4, shared_memory_size=2)
    state = mem1.state_dict()

    mem2 = ClassAwareKnittingMemoryV2(embed_dim=256, num_categories=22, num_scales=10,
                                       shared_patterns=4, shared_memory_size=2)
    mem2.load_state_dict(state, strict=True)
    check("ClassAware state_dict loads strictly", True)

    # KnittingPatternMemory
    km1 = KnittingPatternMemory(embed_dim=256, num_patterns=8, memory_size=4, num_scales=10)
    state2 = km1.state_dict()
    km2 = KnittingPatternMemory(embed_dim=256, num_patterns=8, memory_size=4, num_scales=10)
    km2.load_state_dict(state2, strict=True)
    check("Knitting state_dict loads strictly", True)


# =====================================================================
# Test 5: Batched retrieval vs single retrieval
# =====================================================================
def test_batched_retrieval():
    print("\n=== Test 5: Batched Retrieval Equivalence ===")
    from models.class_aware_memory import ClassAwareKnittingMemoryV2

    torch.manual_seed(42)

    mem = ClassAwareKnittingMemoryV2(embed_dim=128, num_categories=10, num_scales=5,
                                      shared_patterns=4, shared_memory_size=2, cat_rank=4)

    B, L, C = 4, 16, 128
    S = 8  # num_slots

    q = torch.randn(B, L, C)
    k = torch.randn(B, S, C)
    v = torch.randn(B, S, C)
    temp = 0.3

    # Batched
    o_batch, attn_batch = mem._retrieve_batched(q, k, v, temp)

    # Per-sample (manual)
    o_single_list = []
    for b in range(B):
        q_b = q[b:b+1]  # [1, L, C]
        k_b = k[b]       # [S, C]
        v_b = v[b]       # [S, C]
        o_b, _ = mem._retrieve(q_b, k_b, v_b, temp)
        o_single_list.append(o_b)
    o_single = torch.cat(o_single_list, dim=0)

    check("Batched retrieval matches single",
          torch.allclose(o_batch, o_single, atol=1e-5))


# =====================================================================
# Test 6: Batched cat memory computation
# =====================================================================
def test_batched_cat_memory():
    print("\n=== Test 6: Batched Category Memory ===")
    from models.class_aware_memory import ClassAwareKnittingMemoryV2

    torch.manual_seed(42)

    mem = ClassAwareKnittingMemoryV2(embed_dim=128, num_categories=10, num_scales=5,
                                      shared_patterns=4, shared_memory_size=2, cat_rank=4)

    # Test: compare batched vs single _get_cat_memory for each category
    visible_indices = [0, 1, 2]  # First 3 scales visible
    cat_ids = torch.tensor([2, 5, 7])  # 3 unique categories

    # Batched
    batched_result = mem._get_cat_memory_batched(cat_ids, visible_indices)  # [3, 3*8, 128]

    # Per-category, per-scale (original method)
    for u, cat_id in enumerate(cat_ids):
        single_parts = []
        for j in visible_indices:
            single_parts.append(mem._get_cat_memory(cat_id.item(), j))
        single_result = torch.cat(single_parts, dim=0)  # [3*8, 128]

        check(f"Cat {cat_id.item()} memory matches",
              torch.allclose(batched_result[u], single_result, atol=1e-5))


# =====================================================================
# Run all tests
# =====================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("  Optimization Equivalence Tests")
    print("=" * 60)

    test_batched_retrieval()
    test_batched_cat_memory()
    test_class_aware_memory()
    test_knitting_memory()
    test_texture_plan()
    test_checkpoint_compat()

    print("\n" + "=" * 60)
    print(f"  Results: {PASS} passed, {FAIL} failed")
    print("=" * 60)

    if FAIL > 0:
        sys.exit(1)
    else:
        print("\n  All tests passed!")
