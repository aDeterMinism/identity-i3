"""Echo benchmark evaluation for Identity I3 (I3-echo persona, 0 parameters).

I3-echo answers every benchmark prompt by returning it verbatim: f(x) = x.
Most scores are then determined analytically:

  MMLU / HumanEval / BrowseComp : 0.0
      the reference answer is (with probability ~1) not equal to the prompt.

  CNN/DM ROUGE-L                : > 0 (metric artifact)
      the echo overlaps the reference summary. This file can measure it
      empirically on a sample, demonstrating nonzero measured capability
      with zero capability.

Usage:
  python3 echo_benchmark.py                # analytic table only
  python3 echo_benchmark.py --rouge --n 32 # empirical echo ROUGE (needs datasets+rouge_score)
"""
import argparse


def analytic():
    rows = [
        ("MMLU", "0.0", "P(answer == prompt) ~ 0"),
        ("HumanEval", "0.0", "echo passes no test"),
        ("BrowseComp", "0.0", "echo browses nothing"),
        ("CNN/DM ROUGE-L", "> 0", "overlap with reference summary (artifact)"),
        ("Identity-1M (ours)", "100.0", "reference output is x"),
    ]
    w = max(len(r[0]) for r in rows)
    for name, score, note in rows:
        print(f"{name:<{w}}  {score:>6}   {note}")


def rouge(n):
    from datasets import load_dataset
    from rouge_score import rouge_scorer
    ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split=f"test[:{n}]")
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for ex in ds:
        echo = ex["article"][:2000]  # echo, truncated for scorer speed
        s = scorer.score(ex["highlights"], echo)["rougeL"].fmeasure
        scores.append(s)
    print(f"echo ROUGE-L on CNN/DM test[:{n}]: "
          f"mean = {sum(scores) / len(scores):.4f}")
    print("nonzero capability score, zero capability. Q.E.D.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rouge", action="store_true")
    ap.add_argument("--n", type=int, default=32)
    args = ap.parse_args()
    analytic()
    if args.rouge:
        rouge(args.n)
