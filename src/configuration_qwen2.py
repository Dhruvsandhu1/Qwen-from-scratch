"""Qwen2 model configuration"""

try:
    from transformers.configuration_utils import PreTrainedConfig
except Exception:
    # Some HF versions use `PretrainedConfig` (different capitalization)
    from transformers.configuration_utils import PretrainedConfig as PreTrainedConfig
try:
    from transformers.modeling_rope_utils import RopeParameters
except Exception:
    # Minimal fallback dataclass for RopeParameters used only for typing/defaults.
    from dataclasses import dataclass

    @dataclass
    class RopeParameters(dict):
        # Provide dict-like behavior for simple access patterns used in the code.
        rope_type: str = "default"
        rope_theta: float = 10000.0

        def __post_init__(self):
            # Ensure dict-like access
            self.update({"rope_type": self.rope_type, "rope_theta": self.rope_theta})


# Skip `@auto_docstring` here to keep compatibility with different HF versions.
class Qwen2Config(PreTrainedConfig):
    r"""
    Example:

    ```python
    >>> from transformers import Qwen2Model, Qwen2Config

    >>> # Initializing a Qwen2 style configuration
    >>> configuration = Qwen2Config()

    >>> # Initializing a model from the Qwen2-7B style configuration
    >>> model = Qwen2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "qwen2"
    keys_to_ignore_at_inference = ["past_key_values"]

    # Default tensor parallel plan for base model `Qwen2`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    vocab_size: int = 151936
    hidden_size: int = 4096
    intermediate_size: int = 22016
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int | None = 32
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    use_sliding_window: bool = False
    sliding_window: int | None = 4096
    max_window_layers: int = 28
    layer_types: list[str] | None = None
    attention_dropout: float | int = 0.0
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None

    def __post_init__(self, **kwargs):
        self.sliding_window = self.sliding_window if self.use_sliding_window else None
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.layer_types is None:
            self.layer_types = [("sliding_attention" if self.sliding_window is not None and i >= self.max_window_layers else "full_attention") for i in range(self.num_hidden_layers)]

        super().__post_init__(**kwargs)


__all__ = ["Qwen2Config"]
