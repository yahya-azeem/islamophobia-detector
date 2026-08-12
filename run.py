"""Interactive CLI: type your own text, see if it flags as Islamophobic.

Usage:
    python run.py               -> interactive prompt loop
    python run.py "some text"   -> single one-shot check
    python run.py "text1" "text2" -> check multiple strings

Exit interactive mode with Ctrl+C or 'q'.
"""
import os
import sys

import infer

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "model", "student_int8.onnx")


def verdict(p):
    if p >= 0.75:
        return "FLAGGED — Islamophobic (high confidence)"
    if p >= 0.5:
        return "flagged — Islamophobic (moderate)"
    return "clean"


def check_one(text, session, tokenizer):
    p = infer.infer_probs([text], tokenizer=tokenizer, session=session)[0]
    print(f"  [{verdict(p)}]  p={p:.3f}   {text}")
    return p


def main():
    if not os.path.exists(MODEL):
        print("model/student_int8.onnx not found. Run train.py then export_onnx.py first.")
        sys.exit(1)

    session = infer._session()
    tokenizer = infer._tokenizer()

    if len(sys.argv) > 1:
        results = [check_one(a, session, tokenizer) for a in sys.argv[1:]]
        flagged = any(p >= 0.5 for p in results)
        print("\nNote: any text flagged is likely Islamophobic; review borderline cases.")
        sys.exit(0 if not flagged else 2)

    print("Islamophobia detector — type a line and hit Enter. Ctrl+C / 'q' to quit.")
    print("Try mixing: overt hate, dogwhistles (muzzies/mudslime), strawman accusations,\n"
          "false stereotypes, emoji attacks, and genuinely benign Muslim mentions.\n")
    try:
        while True:
            try:
                text = input("> ")
            except EOFError:
                break
            if text.strip().lower() in ("q", "quit", "exit"):
                break
            check_one(text, session, tokenizer)
    except KeyboardInterrupt:
        pass
    print("\nDone.")


if __name__ == "__main__":
    main()