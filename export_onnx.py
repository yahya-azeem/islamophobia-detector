"""
Export the distilled bert-tiny student to ONNX and apply DYNAMIC INT8
quantization (per the plan: "Export to ONNX + dynamic INT8 quantization").

Dynamic quantization keeps activation data (input_ids/attention_mask) dynamic
and quantizes the constant weights to int8, giving a ~4x model-size reduction
with minimal accuracy loss. The quantized model runs in onnxruntime.

Outputs:
  model/student.onnx            fp32 ONNX
  model/student_int8.onnx       dynamic-quantized int8 ONNX
"""
import onnx
import torch
from transformers import AutoTokenizer, BertConfig


def make_student():
    cfg = BertConfig(
        model_type="bert",
        vocab_size=30522,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=512,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        num_labels=2,
    )
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_config(cfg)
    model.load_state_dict(torch.load("model/student_final.bin", weights_only=True))
    return model


def export_fp32(model, tokenizer, out="model/student.onnx"):
    dummy = tokenizer(
        "test input",
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )
    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
        out,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch"},
            "attention_mask": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"exported fp32 -> {out}")


def quantize_dynamic(fp32_path, int8_path):
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_dynamic

    quantize_dynamic(
        model_input=fp32_path,
        model_output=int8_path,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=True,
        extra_options={
            "ActivationSymmetric": True,
            "EnableSubgraph": True,
        },
    )
    print(f"exported int8  -> {int8_path}")


def main():
    model = make_student()
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("model/tokenizer")

    export_fp32(model, tokenizer)
    quantize_dynamic("model/student.onnx", "model/student_int8.onnx")

    for p in ["model/student.onnx", "model/student_int8.onnx"]:
        m = onnx.load(p)
        print(f"{p}: {len(m.graph.node)} nodes, {onnx.checker.check_model.__name__} ok")


if __name__ == "__main__":
    main()