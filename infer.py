"""
Inference entry point: textify_emojis -> tokenize -> ONNX Runtime.

Production path used by the unit tests. The emoji textification MUST be
identical to the preprocessing applied to the training corpus
(data/build_dataset.py) so that "grape" the code word is seen identically
at train and serve time.
"""
import os

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

import emoji as EMOJI

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "model", "student_int8.onnx")
TOKENIZER_DIR = os.path.join(HERE, "model", "tokenizer")
MAX_LEN = 64


_SESSION = None
_TOKENIZER = None


def _session():
    global _SESSION
    if _SESSION is None:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.add_session_config_entry("session.dynamic_block_base", "4")
        _SESSION = ort.InferenceSession(MODEL, opts, providers=["CPUExecutionProvider"])
    return _SESSION


def _tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    return _TOKENIZER


def infer_probs(texts, tokenizer=None, session=None):
    """Return P(toxic) for each string in `texts` (0..1 float)."""
    if session is None:
        session = _session()
    if tokenizer is None:
        tokenizer = _tokenizer()

    clean = [EMOJI.textify_emojis(t) for t in texts]
    inputs = tokenizer(
        clean, return_tensors="np", padding="max_length",
        truncation=True, max_length=MAX_LEN,
    )
    logits = session.run(None, {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64),
    })[0]
    return 1 / (1 + np.exp(-logits[:, 1]))