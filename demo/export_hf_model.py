import torch
from transformers import AutoConfig, Gemma3ForCausalLM
import transformers

MODEL_ID = "google/gemma-3-270m"
# MODEL_ID = "google/gemma-3-1b-it"
# MODEL_ID = "google/gemma-3-4b-it"


def get_hf_model(model_id: str):
    config = AutoConfig.from_pretrained(model_id, attn_implementation="sdpa")
    config.use_cache = True
    # Use the correct AutoModel class for your model architecture
    model = Gemma3ForCausalLM.from_pretrained(model_id, config=config)

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
            [{0: batch, 2: "past_sequence_len"} for _ in range(num_hidden_layers)],
            [{0: batch, 2: "past_sequence_len"} for _ in range(num_hidden_layers)],
        ],
    }
    input_names = [
        "input_ids",
        "attention_mask",
        "position_ids",
        *[f"past_key_values.{i}.key" for i in range(num_hidden_layers)],
        *[f"past_key_values.{i}.value" for i in range(num_hidden_layers)],
    ]
    output_names = [
        "logits",
        *[f"present.{i}.key" for i in range(num_hidden_layers)],
        *[f"present.{i}.value" for i in range(num_hidden_layers)],
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
        past_key_values=make_dynamic_cache(
            [
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
            ]
        ),
    )

    return example_inputs, dynamic_shapes, input_names, output_names


def make_dynamic_cache(
    past_key_values: list[tuple[torch.Tensor, torch.Tensor]],
) -> transformers.cache_utils.DynamicCache:
    cache = transformers.cache_utils.DynamicCache()
    for layer_idx in range(len(past_key_values)):
        key_states, value_states = past_key_values[layer_idx]
        cache.update(key_states, value_states, layer_idx)
    return cache


from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


class TextGenerationModelWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

        # This is the same as sdpa, but mask creation does not use `vmap` which is not exportable
        ALL_MASK_ATTENTION_FUNCTIONS.register(
            "sdpa_without_vmap", sdpa_mask_without_vmap
        )
        ALL_ATTENTION_FUNCTIONS.register(
            "sdpa_without_vmap", ALL_ATTENTION_FUNCTIONS["sdpa"]
        )
        self.model.model.config._attn_implementation = "sdpa_without_vmap"

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
            past_key_values=past_key_values,
            use_cache=True,
        )
        return hf_output.logits, hf_output.past_key_values


model, config = get_hf_model(MODEL_ID)
# Wrap the model to adjust the forward method for ONNX export
model = TextGenerationModelWrapper(model)
# Obtain example inputs and dynamic axes
example_kwargs, dynamic_shapes, input_names, output_names = (
    create_text_gen_example_inputs(config)
)

# transformers.integrations.executorch.register_dynamic_cache_export_support()

# ONNX Export
# Disable fake tensor cache to avoid issues vmap
# with torch._dynamo.config.patch(fake_tensor_cache_enabled=False):
onnx_program = torch.onnx.export(
    model,
    (),
    kwargs=example_kwargs,
    input_names=input_names,
    output_names=output_names,
    dynamic_shapes=dynamic_shapes,
    opset_version=23,
    dynamo=True,
    # report=True,  # Uncomment to get a report of the export
)

print("✅ Export successful")

onnx_program.save("gemma-3-270m.onnx", external_data=True)

print("🧠 Model saved")
