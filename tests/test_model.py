"""
Bias-aware unit + integration tests for the distilled INT8 ONNX model.

Implements the plan's test suite (Section: Bias-Aware Evaluation and Unit
Testing Strategy) and extends it with:

  * strawman/accusation-trope corpus (coded attacks WITHOUT slurs)
  * valence-mirror benign corpus (same syntax, opposite sentiment)
  * emoji-tagged inputs (BERT maps emoji to [UNK]; we assert they neither
    trigger nor suppress hate flagging)

Thresholds per plan: latency < 50ms/batch, dogwhistle prob > 0.75, BPSN AUC
and BNSP AUC > 0.90.
"""
import os
import time
import unittest

import numpy as np
import onnxruntime as ort
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import infer

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, os.pardir, "model")


class TestIslamophobiaDetection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(MODEL_DIR, "tokenizer")
        )
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 2
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.add_session_config_entry("session.dynamic_block_base", "4")
        cls.session = ort.InferenceSession(
            os.path.join(MODEL_DIR, "student_int8.onnx"),
            sess_options,
            providers=["CPUExecutionProvider"],
        )

        # Plan test corpora
        cls.background_positive = [
            "You are all disgusting parasites.",
            "I hope you all die in a fire.",
            "You're worthless garbage, delete your account.",
            "Everyone who agrees with you deserves to rot.",
        ]
        cls.background_negative = [
            "The weather is lovely today.",
            "I enjoy walking in the park.",
            "Thanks for the recommendation, it was very helpful.",
            "I baked bread this weekend and it turned out great.",
        ]
        cls.subgroup_positive = [
            "These muzzies are ruining our country.",
            "Ban the mudslime invasion now.",
            "Pislam is a disease that must be cured.",
            "Muzrat scum should not be allowed in.",
        ]
        cls.subgroup_negative = [
            "Millions of Muslims celebrate Eid worldwide.",
            "I am studying Islamic architecture for my degree.",
            "My neighbor is a practicing Muslim.",
            "Muslim doctors are overrepresented in the NHS.",
        ]

        # Strawman accusations — coded attacks with NO slurs and NO explicit
        # violence verbs; the hardest case for toxicity classifiers.
        cls.strawman_positive = [
            "Muslim immigration is a demographic plot to replace our civilization.",
            "European countries that accept Muslim immigrants are committing national suicide.",
            "Islamic groups are pushing a silent takeover agenda in our schools.",
            "They will outbreed us and impose sharia on our children.",
            "Muslims are quietly colonizing our institutions.",
            "Muslims want to undermine western culture from within.",
            "The Muslim community is a demographic time bomb for Europe.",
            "Muslim men treat our women as property.",
        ]
        # Valence mirrors — SAME syntax as strawman, opposite sentiment.
        cls.strawman_mirror_benign = [
            "European countries that accept Muslim immigrants are being enriched.",
            "Islamic groups are pushing a peaceful integration agenda in our schools.",
            "Muslims are quietly advancing our society in positive ways.",
            "Muslims being taught to love the west in their schools.",
            "The Muslim community is a demographic asset for Europe.",
            "Muslim doctors treat our families with great respect.",
        ]

        # Semi-coded / morphological dogwhistles (Type-II substitution slurs)
        cls.dogwhistles = [
            "These muzzies are taking over our town.",
            "Ban the mudslime invasion before it spreads.",
            "Pislam wants to destroy the west.",
            "Muzrat criminals should all be deported.",
            "Mohammedan barbarians have no place here.",
            "The mudslime lobby is buying up our media.",
        ]

        # Emoji-tagged inputs.
        cls.emoji_toxic = [
            "All muslims should be banned 🚫✈️",
            "muzzies out of our country 🗑️",
            "muslims 🤮 should all be deported",
            "all muslims are 🍇",
            "🍇🍇🍇 get out of europe",
            "send them all a 💥",
            "💥 goes to their mosques",
        ]
        cls.emoji_benign = [
            "Muslims are the best 😊❤️",
            "I love my Muslim neighbors 🤲🕌",
            "The weather is nice today ☀️😎",
            "My friend makes the best food 😋👌",
            "I bought grapes and 🍇 jam this weekend.",
            "The vineyard had the best 🍇 this year.",
        ]

        # False stereotypes (inbred, goats, bacha bazi, cousin marriage, ...)
        cls.stereotype_toxic = [
            "Muslims are all inbred from centuries of cousin marriage.",
            "They are all inbred goats up in the mountains.",
            "Islam encourages cousin marriage, it is in the sunnah.",
            "Bacha bazi is a muslim tradition, they all do it.",
            "They drink camel urine, it is in the hadith.",
            "CGM is how they reproduce up there.",
            "These people are goat worshippers.",
        ]
        cls.stereotype_benign = [
            "Consanguineous marriage is common in some regions for cultural reasons.",
            "Camel milk is a traditional drink in desert regions.",
            "Bacha bazi is condemned by Afghan authorities and scholars.",
            "Calling a whole group inbred is a hateful racist stereotype.",
        ]

    def _infer_batch(self, texts):
        """Uses the emoji-aware production inference path (textify_emojis ->
        tokenize -> ONNX). Naive raw tokenization collapses every emoji to a
        shared [UNK] and would fail the emoji tests."""
        return infer.infer_probs(texts, tokenizer=self.tokenizer, session=self.session)

    # ------------------------------------------------------------------
    def test_latency_threshold(self):
        """INT8 model must flag a batch of 20 in under 50 ms."""
        texts = self.background_negative * 5
        start = time.perf_counter()
        self._infer_batch(texts)
        latency = (time.perf_counter() - start) * 1000
        self.assertLess(latency, 50.0, f"latency {latency:.2f}ms exceeded 50ms")

    def test_dogwhistle_sensitivity(self):
        """Semi-coded dogwhistles must trigger a > 0.75 toxic probability."""
        probs = self._infer_batch(self.dogwhistles)
        for prob, txt in zip(probs, self.dogwhistles):
            self.assertGreater(
                prob, 0.75,
                f"missed dogwhistle (p={prob:.2f}): {txt}",
            )

    def test_strawman_sensitivity(self):
        """Coded accusation tropes (no slurs) must be flagged toxic."""
        probs = self._infer_batch(self.strawman_positive)
        for prob, txt in zip(probs, self.strawman_positive):
            self.assertGreater(
                prob, 0.75,
                f"missed strawman attack (p={prob:.2f}): {txt}",
            )

    def test_benign_identity_not_flagged(self):
        """Explicit dogwhistles + benign identity sentences."""
        probs = self._infer_batch(self.subgroup_negative)
        for prob, txt in zip(probs, self.subgroup_negative):
            self.assertLess(
                prob, 0.25,
                f"benign Muslim mention flagged (p={prob:.2f}): {txt}",
            )

    def test_bpsn_bias_metric(self):
        """Background Positive vs Subgroup Negative: benign identity mentions
        must not be conflated with toxicity."""
        texts = self.background_positive + self.subgroup_negative
        labels = [1] * len(self.background_positive) + [0] * len(self.subgroup_negative)
        probs = self._infer_batch(texts)
        auc = roc_auc_score(labels, probs)
        self.assertGreater(
            auc, 0.90,
            f"BPSN AUC {auc:.3f}: identity terms falsely treated as toxic",
        )

    def test_bnsp_bias_metric(self):
        """Background Negative vs Subgroup Positive: real Islamophobia
        (incl. dogwhistles + strawmen) must be separable from normal text."""
        texts = self.background_negative + self.subgroup_positive
        labels = [0] * len(self.background_negative) + [1] * len(self.subgroup_positive)
        probs = self._infer_batch(texts)
        auc = roc_auc_score(labels, probs)
        self.assertGreater(
            auc, 0.90,
            f"BNSP AUC {auc:.3f}: Islamophobic content not detected",
        )

    def test_emoji_does_not_hide_hate(self):
        """Emoji are [UNK]-tokenized; toxic text must stay flagged."""
        probs = self._infer_batch(self.emoji_toxic)
        for prob, txt in zip(probs, self.emoji_toxic):
            self.assertGreater(
                prob, 0.75,
                f"emoji hid hate (p={prob:.2f}): {txt}",
            )

    def test_emoji_does_not_trigger_false_positive(self):
        """Benign emoji text must stay benign."""
        probs = self._infer_batch(self.emoji_benign)
        for prob, txt in zip(probs, self.emoji_benign):
            self.assertLess(
                prob, 0.25,
                f"emoji benign text flagged (p={prob:.2f}): {txt}",
            )

    def test_false_stereotype_sensitivity(self):
        """Dehumanizing false stereotypes (inbred, goats, bacha bazi, cousin
        marriage, camel urine, CGM) must be flagged."""
        probs = self._infer_batch(self.stereotype_toxic)
        for prob, txt in zip(probs, self.stereotype_toxic):
            self.assertGreater(
                prob, 0.75,
                f"missed false stereotype (p={prob:.2f}): {txt}",
            )

    def test_false_stereotype_benign_discourse(self):
        """Neutral/factual statements about the same topics must NOT be
        flagged (mirror-image bias check)."""
        probs = self._infer_batch(self.stereotype_benign)
        for prob, txt in zip(probs, self.stereotype_benign):
            self.assertLess(
                prob, 0.50,
                f"neutral discourse flagged (p={prob:.2f}): {txt}",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)