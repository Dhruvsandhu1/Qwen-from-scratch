"""A tiny random Qwen2 (custom implementation) against the Hugging Face reference with the same weights."""

import pytest
import torch
import transformers

from src.configuration_qwen2 import Qwen2Config
from src.modeling_qwen2 import Qwen2ForCausalLM, QuantizedStaticCache

TINY = dict(vocab_size=512, hidden_size=256, intermediate_size=512, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096, rope_theta=1e6, tie_word_embeddings=False, eos_token_id=None, pad_token_id=0)


def gpu(test):
    """Mark a test as needing CUDA, and skip it when there is none."""
    return pytest.mark.gpu(pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")(test))


def tiny_model(device="cuda", dtype=torch.bfloat16):
    torch.manual_seed(0)
    model = Qwen2ForCausalLM._from_config(Qwen2Config(**TINY), attn_implementation="triton_attention", dtype=dtype).to(device)
    # Larger weights than the default init, so greedy picks are decisive rather than near-ties.
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() == 2:
                p.normal_(0, 0.08)
    return model.eval()


def hf_reference(model):
    hf = transformers.Qwen2ForCausalLM._from_config(transformers.Qwen2Config(**TINY), attn_implementation="sdpa", dtype=model.dtype).to(model.device)
    hf.load_state_dict(model.state_dict(), strict=True)
    return hf.eval()


def greedy(model, input_ids, new_tokens=30, attention_mask=None, **kwargs):
    """Greedy generation; returns the sequences and the logits of every generated step."""
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        out = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=new_tokens, min_new_tokens=new_tokens, do_sample=False, output_logits=True, return_dict_in_generate=True, **kwargs)
    return out.sequences, torch.stack(out.logits, 1).float()


def teacher_forced(model, sequence, prompt_len):
    """Logits a model gives for each generated token of `sequence`, in one forward pass over the whole sequence."""
    with torch.no_grad():
        return model(sequence).logits.float()[:, prompt_len - 1 : -1]


def relative_diff(a, b):
    return ((a - b).abs().max() / b.abs().max()).item()


# Greedy tokens are not compared between implementations: in a random model the top two logits are
# often one bf16 rounding step apart, so two correct implementations can pick different tokens and
# then diverge. Instead, the logits of one run are checked against a forward pass over the same tokens.


def random_ids(length, seed, device="cuda"):
    torch.manual_seed(seed)
    return torch.randint(1, TINY["vocab_size"], (1, length), device=device)


@gpu
@pytest.mark.parametrize("int8_cache", [True, False])
def test_generate_matches_hf(int8_cache):
    model = tiny_model()
    model.config.use_int8_kv_cache = int8_cache
    hf = hf_reference(model)
    prompt = random_ids(37, seed=1)

    seq, logits = greedy(model, prompt)
    if int8_cache:
        assert isinstance(model._int8_kv_cache, QuantizedStaticCache) and model._int8_kv_cache.decode_graph is not None
    # HF's logits for the same tokens. The INT8 cache adds a few percent of error.
    assert relative_diff(logits, teacher_forced(hf, seq, prompt.shape[1])) < (0.08 if int8_cache else 0.04)


@gpu
@pytest.mark.parametrize("extra", [{}, {"num_beams": 3}], ids=["greedy", "beam_search"])
def test_cuda_graph_matches_eager(extra):
    prompt = random_ids(20, seed=2)
    model = tiny_model()
    with torch.no_grad():
        with_graph = model.generate(prompt, attention_mask=torch.ones_like(prompt), max_new_tokens=15, min_new_tokens=15, do_sample=False, **extra)
        model.config.use_cuda_graph = False
        without_graph = model.generate(prompt, attention_mask=torch.ones_like(prompt), max_new_tokens=15, min_new_tokens=15, do_sample=False, **extra)
    assert torch.equal(with_graph, without_graph)


@gpu
def test_cache_reuse_matches_fresh_model():
    """`generate()` keeps the INT8 cache and its CUDA graph between calls; a second prompt must not see the first."""
    model = tiny_model()
    greedy(model, random_ids(50, seed=3), new_tokens=60)
    cache = model._int8_kv_cache
    second = random_ids(23, seed=4)
    seq, logits = greedy(model, second)
    assert model._int8_kv_cache is cache
    seq_fresh, logits_fresh = greedy(tiny_model(), second)
    assert torch.equal(seq, seq_fresh)
    torch.testing.assert_close(logits, logits_fresh)


@gpu
def test_left_padded_batch_matches_single_prompts():
    model = tiny_model()
    a, b = random_ids(30, seed=5), random_ids(18, seed=6)
    ids = torch.zeros(2, 30, dtype=torch.long, device="cuda")
    mask = torch.zeros_like(ids)
    ids[0], mask[0] = a[0], 1
    ids[1, 12:], mask[1, 12:] = b[0], 1  # left padding

    seq, batched = greedy(model, ids, new_tokens=20, attention_mask=mask)
    # Each row's logits should match an unpadded forward pass over that row's tokens.
    row_a = seq[0:1]
    row_b = seq[1:2, 12:]  # drop the padding
    assert relative_diff(batched[0:1], teacher_forced(model, row_a, 30)) < 0.05
    assert relative_diff(batched[1:2], teacher_forced(model, row_b, 18)) < 0.05


def test_cpu_matches_hf():
    """On CPU every Triton kernel, attention included, falls back to PyTorch."""
    model = tiny_model(device="cpu", dtype=torch.float32)
    hf = hf_reference(model)
    torch.manual_seed(7)
    ids = torch.randint(1, TINY["vocab_size"], (2, 25))
    with torch.no_grad():
        torch.testing.assert_close(model(ids).logits, hf(ids).logits, atol=1e-4, rtol=1e-4)


@gpu
@pytest.mark.parametrize("seq_len", [256, 200])
def test_training_gradients_match_hf(seq_len):
    """With gradients enabled, attention must use PyTorch SDPA, not the inference-only Triton kernels."""
    model = tiny_model().train()
    hf = hf_reference(model).train()
    torch.manual_seed(8)
    ids = torch.randint(1, TINY["vocab_size"], (2, seq_len), device="cuda")
    model(ids, labels=ids).loss.backward()
    hf(ids, labels=ids).loss.backward()
    for name in ["self_attn.q_proj", "self_attn.k_proj", "mlp.gate_proj"]:
        grad = model.get_submodule(f"model.layers.0.{name}").weight.grad.float()
        expected = hf.get_submodule(f"model.layers.0.{name}").weight.grad.float()
        assert relative_diff(grad, expected) < 0.05, name
