"""Script to export a Hugging Face text model to ONNX format using torch.onnx.export.

Verified with transformers==4.55
"""

from __future__ import annotations

import os

import torch
from transformers import AutoConfig, AutoModelForCausalLM
import transformers

from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from onnxscript.rewriter.ort_fusions import optimize_for_ort


# MODEL_ID = "google/gemma-3-270m-it"
MODEL_ID = "google/gemma-3-1b-it"
# MODEL_ID = "google/gemma-3-4b-it"
# MODEL_ID = "google/gemma-3-27b-it"

MODEL_NAME = MODEL_ID.split("/")[-1]


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def sdpa_attention_forward_with_check(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    if hasattr(module, "num_key_value_groups"):
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)
    if attention_mask is not None and attention_mask.ndim == 4:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]
        torch._check(
            attention_mask.shape[-1] == key.shape[-2],
            lambda: "attention_mask.shape[-1] == key.shape[-2] should be true",
        )

    if is_causal is None:
        is_causal = (
            query.shape[2] > 1
            and attention_mask is None
            and getattr(module, "is_causal", True)
        )

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=is_causal,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, None


# Patch the attention functions to use the custom SDPA without vmap
# This is the same as sdpa, but mask creation does not use `vmap` which is not exportable
ALL_MASK_ATTENTION_FUNCTIONS.register(
    "sdpa_without_vmap",
    transformers.integrations.executorch.sdpa_mask_without_vmap,
)
ALL_ATTENTION_FUNCTIONS.register("sdpa_without_vmap", sdpa_attention_forward_with_check)


def get_hf_model(model_id: str):
    """Load a Hugging Face model and its config."""
    config = AutoConfig.from_pretrained(
        model_id, attn_implementation="sdpa_without_vmap"
    )
    config.use_cache = True
    # MARK: Use the correct AutoModel class for your model architecture
    model = AutoModelForCausalLM.from_pretrained(model_id, config=config)

    return model, config


def create_text_gen_example_inputs(
    config, batch_size: int = 2, seq_len: int = 3, past_seq_len: int = 2
):
    """Create example inputs and dynamic axes for ONNX export."""
    config = config.get_text_config()
    num_hidden_layers = config.num_hidden_layers
    # batch = "batch"
    # sequence_len = "sequence_len"
    # past_sequence_len = "past_sequence_len"
    batch = torch.export.Dim("batch")
    sequence_len = torch.export.Dim("sequence_len")
    # past_sequence_len = torch.export.Dim("past_sequence_len")

    dynamic_shapes = {
        "input_ids": {0: batch, 1: sequence_len},
        "attention_mask": {
            0: batch,
            1: "past_sequence_len+sequence_len",
        },
        "position_ids": {
            0: batch,
            1: sequence_len,
        },
        "past_key_values": [
            ({0: batch, 2: "past_sequence_len"}, {0: batch, 2: "past_sequence_len"})
            for _ in range(num_hidden_layers)
        ],
    }
    past_key_values_names = []
    for i in range(num_hidden_layers):
        past_key_values_names.append(f"past_key_values.{i}.key")
        past_key_values_names.append(f"past_key_values.{i}.value")
    input_names = [
        "input_ids",
        "attention_mask",
        "position_ids",
        *past_key_values_names,
    ]

    present_names = []
    for i in range(num_hidden_layers):
        present_names.append(f"present.{i}.key")
        present_names.append(f"present.{i}.value")
    output_names = [
        "logits",
        *present_names,
    ]

    num_key_value_heads = config.num_key_value_heads
    head_dim = config.head_dim

    example_inputs = dict(
        input_ids=torch.randint(0, 2, (batch_size, seq_len), dtype=torch.int64),
        attention_mask=torch.ones(
            (batch_size, past_seq_len + seq_len),
            dtype=torch.int64,
        ),
        position_ids=torch.arange(
            past_seq_len,
            past_seq_len + seq_len,
            dtype=torch.int64,
        ).expand((batch_size, -1)),
        past_key_values=[
            (
                torch.randn(
                    batch_size,
                    num_key_value_heads,
                    seq_len,
                    head_dim,
                ),
                torch.randn(
                    batch_size,
                    num_key_value_heads,
                    seq_len,
                    head_dim,
                ),
            )
            for _ in range(num_hidden_layers)
        ],
    )

    return example_inputs, dynamic_shapes, input_names, output_names


def unpack_dynamic_cache(
    dynamic_cache: transformers.cache_utils.DynamicCache,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Unpack a DynamicCache into past_key_values format."""
    return [(layer.keys, layer.values) for layer in dynamic_cache.layers]


def make_dynamic_cache(
    past_key_values: list[tuple[torch.Tensor, torch.Tensor]],
) -> transformers.cache_utils.DynamicCache:
    """Create a DynamicCache from past_key_values."""
    cache = transformers.cache_utils.DynamicCache()
    for layer_idx in range(len(past_key_values)):
        key_states, value_states = past_key_values[layer_idx]
        cache.update(key_states, value_states, layer_idx)
    return cache


class TextGenerationModelWrapper(torch.nn.Module):
    """A wrapper around a Hugging Face model to adjust the forward method for ONNX export."""

    def __init__(self, model: transformers.PreTrainedModel):
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids,
        attention_mask,
        position_ids,
        past_key_values,
    ):
        hf_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=make_dynamic_cache(past_key_values),
            use_cache=True,
        )
        return hf_output.logits, unpack_dynamic_cache(hf_output.past_key_values)


model, config = get_hf_model(MODEL_ID)

# Wrap the model to adjust the forward method for ONNX export
model = TextGenerationModelWrapper(model)

# Obtain example inputs and dynamic axes
example_kwargs, dynamic_shapes, input_names, output_names = (
    create_text_gen_example_inputs(config)
)

# ONNX Export
onnx_program = torch.onnx.export(
    model,
    (),
    kwargs=example_kwargs,
    input_names=input_names,
    output_names=output_names,
    dynamic_shapes=dynamic_shapes,
    opset_version=20,  # Set to 20 for ORT fusion rules
    dynamo=True,
    # report=True,  # Uncomment to get a report of the export
)

print("✅ Export successful")

print("Optimize the model with ONNX Runtime custom ops...")

_, count = optimize_for_ort(onnx_program.model)
print(f"Applied optimizations: {count}")

os.makedirs(f"models/{MODEL_NAME}_ort", exist_ok=True)

# Use the ONNXProgram.save method to save the model. Specifying external_data=True
# will save the model weights in external files, which is required for models > 2GB
onnx_program.save(f"models/{MODEL_NAME}_ort/{MODEL_NAME}.onnx", external_data=True)

print(f"🧠 Model saved to models/{MODEL_NAME}_ort/{MODEL_NAME}.onnx")
