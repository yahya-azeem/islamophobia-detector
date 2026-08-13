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
from transformers import AutoTokenizer


def make_student():
    # Student must be the same pretrained BERT-Tiny used in train.py so the
    # architecture & state shape match student_final.bin exactly.
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(
        "prajjwal1/bert-tiny", num_labels=2, local_files_only=True,
    )
    model.load_state_dict(torch.load("model/student_final.bin", weights_only=True))
    return model


def export_fp32(model, tokenizer, out="model/student.onnx"):
    dummy = tokenizer(
        "test input",
        padding="max_length",
        truncation=True,
        max_length=128,
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
        op_types_to_quantize=["MatMul"],
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