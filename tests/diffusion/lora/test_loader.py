import torch
import torch.nn as nn
from torch.testing import assert_close

from vllm_omni.diffusion.lora.loader import (
    LoraLoaderMixin,
    _prepare_lora_delta,
    _remap_state_dict_keys,
)

HEAD_DIM = 64
RANK = 32
NUM_LAYERS = 10


class DummyTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        bias: bool = False,
    ):
        super().__init__()
        self.attn = nn.Module()
        self.attn.to_q = nn.Linear(d_model, d_model, bias=bias)
        self.attn.to_k = nn.Linear(d_model, d_model, bias=bias)
        self.attn.to_v = nn.Linear(d_model, d_model, bias=bias)
        self.attn.to_out = nn.Sequential(nn.Linear(d_model, d_model, bias=bias))


class DummyTransformer(nn.Module):
    def __init__(self, num_layers: int, d_model: int, bias: bool = False):
        super().__init__()

        self.blocks = nn.ModuleList([DummyTransformerBlock(d_model, bias=bias) for _ in range(num_layers)])


class DummyTransformerWithStackedParams(DummyTransformer):
    def __init__(self, num_layers: int, d_model: int, bias: bool = False):
        super().__init__(num_layers, d_model, bias=bias)
        self.stacked_params_mapping = [
            (".attn.to_qkv", ".attn.to_q", "q"),
            (".attn.to_qkv", ".attn.to_k", "k"),
            (".attn.to_qkv", ".attn.to_v", "v"),
        ]


class DummyPipeline(nn.Module, LoraLoaderMixin):
    def __init__(self, num_layers: int, d_model: int, bias: bool = False, stacked_params: bool = False):
        super().__init__()
        if stacked_params:
            self.transformer = DummyTransformerWithStackedParams(num_layers, d_model, bias=bias)
        else:
            self.transformer = DummyTransformer(num_layers, d_model, bias=bias)


# ======================================================
# helper functions for test
# ======================================================


def make_dummy_lora_state_dict(
    base_key,
    weight_a: torch.Tensor | None = None,
    weight_b: torch.Tensor | None = None,
    weight_bias: torch.Tensor | None = None,
    bias: bool = False,
    lora_a_suffix: str = "lora_A.weight",
    lora_b_suffix: str = "lora_B.weight",
    lora_bias_suffix: str = "bias",
):
    """
    Create a dummy LoRA state dict.

    Returns:
        tuple: (state_dict, expected_delta)
    """
    key_a = f"{base_key}.{lora_a_suffix}"
    key_b = f"{base_key}.{lora_b_suffix}"
    key_bias = f"{base_key}.{lora_bias_suffix}" if bias else None

    if weight_a is None:
        weight_a = torch.randn(RANK, HEAD_DIM)
    if weight_b is None:
        weight_b = torch.randn(HEAD_DIM, RANK)

    state_dict = {}
    expected_delta = None

    if bias:
        state_dict[key_a] = weight_a
        state_dict[key_b] = weight_b
        weight_bias = torch.randn(HEAD_DIM)
        state_dict[key_bias] = weight_bias
        expected_delta = weight_bias
    else:
        state_dict[key_a] = weight_a
        state_dict[key_b] = weight_b
        expected_delta = weight_b @ weight_a

    return state_dict, expected_delta


def make_lora_state_dict_for_module(
    module,
    lora_rank=RANK,
    bias=False,
    lora_a_suffix: str = "lora_A.weight",
    lora_b_suffix: str = "lora_B.weight",
    lora_bias_suffix: str = "bias",
):
    state_dict = {}
    for name, param in module.named_parameters():
        if name.endswith(".weight"):
            base_key = name[: -len(".weight")]
            d_in, d_out = param.shape
            weight_a = torch.randn(lora_rank, d_out)
            weight_b = torch.randn(d_in, lora_rank)
            key_a = f"{base_key}.{lora_a_suffix}"
            key_b = f"{base_key}.{lora_b_suffix}"
            state_dict[key_a] = weight_a
            state_dict[key_b] = weight_b
        elif name.endswith(".bias") and bias:
            base_key = name[: -len(".bias")]
            key_bias = f"{base_key}.{lora_bias_suffix}"
            state_dict[key_bias] = torch.randn_like(param)

    return state_dict


# ======================================================
# Test loader helper functions
# ======================================================
def test_prepare_lora_delta():
    base_key = "transformer.attn.to_q"
    lora_state_dict, expected_delta = make_dummy_lora_state_dict(base_key)

    actual_delta, used_keys = _prepare_lora_delta(
        lora_state_dict,
        base_key,
    )

    # validate the delta
    assert actual_delta is not None
    assert_close(actual_delta, expected_delta)

    # validate the accessed keys
    expected_key_a = f"{base_key}.lora_A.weight"
    expected_key_b = f"{base_key}.lora_B.weight"
    assert expected_key_a in used_keys
    assert expected_key_b in used_keys


def test_prepare_lora_delta_with_bias():
    base_key = "transformer.attn.to_q"
    weight_bias = torch.randn(HEAD_DIM)
    lora_state_dict, expected_delta = make_dummy_lora_state_dict(base_key, weight_bias=weight_bias, bias=True)

    actual_delta, used_keys = _prepare_lora_delta(
        lora_state_dict,
        base_key,
        is_bias=True,
    )

    # validate the delta
    assert actual_delta is not None
    assert_close(expected_delta, actual_delta)

    # validate the accessed keys
    expected_key_bias = f"{base_key}.bias"
    assert expected_key_bias in used_keys


def test_prepare_lora_delta_with_stacked_params():
    base_key = "transformer.attn.to_qkv"

    sd0, delta0 = make_dummy_lora_state_dict("transformer.attn.to_q")
    sd1, delta1 = make_dummy_lora_state_dict("transformer.attn.to_k")
    sd2, delta2 = make_dummy_lora_state_dict("transformer.attn.to_v")

    lora_state_dict = {}
    lora_state_dict.update(sd0)
    lora_state_dict.update(sd1)
    lora_state_dict.update(sd2)
    expected_delta = torch.concat([delta0, delta1, delta2])

    stacked_mapping = {".attn.to_qkv": [".attn.to_q", ".attn.to_k", ".attn.to_v"]}

    actual_delta, used_keys = _prepare_lora_delta(
        lora_state_dict,
        base_key,
        stacked_mapping,
    )

    # validate the delta
    assert actual_delta is not None
    assert_close(actual_delta, expected_delta)

    # validate the used keys
    key_q_a = "transformer.attn.to_q.lora_A.weight"
    key_q_b = "transformer.attn.to_q.lora_B.weight"
    key_k_a = "transformer.attn.to_k.lora_A.weight"
    key_k_b = "transformer.attn.to_k.lora_B.weight"
    key_v_a = "transformer.attn.to_v.lora_A.weight"
    key_v_b = "transformer.attn.to_v.lora_B.weight"

    for key in [key_q_a, key_q_b, key_k_a, key_k_b, key_v_a, key_v_b]:
        assert key in used_keys


def test_prepare_lora_delta_with_stacked_params_with_bias():
    base_key = "transformer.attn.to_qkv"
    sd0, delta0 = make_dummy_lora_state_dict("transformer.attn.to_q", bias=True)
    sd1, delta1 = make_dummy_lora_state_dict("transformer.attn.to_k", bias=True)
    sd2, delta2 = make_dummy_lora_state_dict("transformer.attn.to_v", bias=True)

    lora_state_dict = {}
    lora_state_dict.update(sd0)
    lora_state_dict.update(sd1)
    lora_state_dict.update(sd2)
    expected_delta = torch.concat([delta0, delta1, delta2])

    stacked_mapping = {".attn.to_qkv": [".attn.to_q", ".attn.to_k", ".attn.to_v"]}

    actual_delta, used_keys = _prepare_lora_delta(
        lora_state_dict,
        base_key,
        stacked_mapping,
        is_bias=True,
    )

    # validate the delta
    assert actual_delta is not None
    assert_close(expected_delta, actual_delta)

    # validate the used keys
    key_q_bias = "transformer.attn.to_q.bias"
    key_k_bias = "transformer.attn.to_k.bias"
    key_v_bias = "transformer.attn.to_v.bias"
    for key in [key_q_bias, key_k_bias, key_v_bias]:
        assert key in used_keys


def test_remap_state_dict_keys():
    lora_state_dict, _ = make_dummy_lora_state_dict("transformer.attn.to_out.0.proj")

    lora_state_dict = _remap_state_dict_keys(
        lora_state_dict,
        [(".to_out.0.", ".to_out.")],
    )

    assert len(lora_state_dict) == 2
    assert "transformer.attn.to_out.proj.lora_A.weight" in lora_state_dict
    assert "transformer.attn.to_out.proj.lora_B.weight" in lora_state_dict


def test_remap_state_dict_keys_matched_two():
    lora_state_dict, _ = make_dummy_lora_state_dict(
        "transformer.attn.to_out.0", bias=True, lora_bias_suffix="lora_B.bias"
    )

    lora_state_dict = _remap_state_dict_keys(
        lora_state_dict,
        [(".to_out.0.", ".to_out."), (".lora_B.bias", ".bias")],
    )

    assert len(lora_state_dict) == 3
    assert "transformer.attn.to_out.bias" in lora_state_dict


# ======================================================
# Test LoraLoaderMixin
# ======================================================
class TestLoraLoaderMixin:
    def test_load_lora_into_module(self):
        pipeline = DummyPipeline(NUM_LAYERS, HEAD_DIM)
        original_weights = {name: param.clone() for name, param in pipeline.transformer.named_parameters()}

        lora_state_dict = make_lora_state_dict_for_module(pipeline)
        # validate the state dict is valid
        assert len(lora_state_dict) > 0

        used_keys = pipeline.load_lora_into_module(lora_state_dict, pipeline.transformer)

        # validate the weights are updated
        for name, param in pipeline.transformer.named_parameters():
            if name.endswith(".weight"):
                assert not torch.allclose(param, original_weights[name])

        # validate all keys are used
        missing_keys = set(lora_state_dict.keys()) - used_keys
        assert len(missing_keys) == 0

    def test_load_lora_into_module_bias(self):
        pipeline = DummyPipeline(NUM_LAYERS, HEAD_DIM, bias=True)
        original_weights = {name: param.clone() for name, param in pipeline.transformer.named_parameters()}

        lora_state_dict = make_lora_state_dict_for_module(pipeline, bias=True)
        # validate the state dict is valid
        assert len(lora_state_dict) > 0

        used_keys = pipeline.load_lora_into_module(lora_state_dict, pipeline.transformer)

        # validate the weights are updated
        for name, param in pipeline.transformer.named_parameters():
            if name.endswith(".weight"):
                assert not torch.allclose(param, original_weights[name])

        # validate all keys are used
        missing_keys = set([key for key in lora_state_dict.keys() if key.endswith(".bias")]) - used_keys
        assert len(missing_keys) == 0

    def test_load_lora_into_module_stacked_params(self):
        pipeline = DummyPipeline(NUM_LAYERS, HEAD_DIM, stacked_params=True)
        original_weights = {name: param.clone() for name, param in pipeline.transformer.named_parameters()}

        lora_state_dict = make_lora_state_dict_for_module(pipeline)
        # validate the state dict is valid
        assert len(lora_state_dict) > 0

        used_keys = pipeline.load_lora_into_module(lora_state_dict, pipeline.transformer)

        # validate the weights are updated
        for name, param in pipeline.transformer.named_parameters():
            if name.endswith(".weight"):
                assert not torch.allclose(param, original_weights[name])

        # validate all keys are used
        missing_keys = set(lora_state_dict.keys()) - used_keys
        assert len(missing_keys) == 0

    def test_load_lora_into_module_stacked_params_with_bias(self):
        pipeline = DummyPipeline(NUM_LAYERS, HEAD_DIM, bias=True, stacked_params=True)
        original_weights = {name: param.clone() for name, param in pipeline.transformer.named_parameters()}

        lora_state_dict = make_lora_state_dict_for_module(pipeline, bias=True)
        # validate the state dict is valid
        assert len(lora_state_dict) > 0

        used_keys = pipeline.load_lora_into_module(lora_state_dict, pipeline.transformer)

        # validate the weights are updated
        for name, param in pipeline.transformer.named_parameters():
            if name.endswith(".weight"):
                assert not torch.allclose(param, original_weights[name])

        # validate all keys are used
        missing_keys = set([key for key in lora_state_dict.keys() if key.endswith(".bias")]) - used_keys
        assert len(missing_keys) == 0
