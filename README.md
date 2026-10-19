<div align="center">
  <h1>Identity I3</h1>
  <p><b>Open Frontier Weights, Zero Frontier Intelligence</b></p>
</div>

## 1. Model Introduction

Identity I3 is an open-weight, native unimodal model, and our least capable
model to date. It is a 2.25-trillion-parameter model built on a single linear
layer. Trained from random initialization on an infinite synthetic corpus of
(x, x) pairs — with adversarial negative samples (x, ỹ) for robustness —
Identity I3 learns to return its input. All of it. Exactly.

Key features:

- **New Architecture**: a matrix.
- **Long-Horizon Identity**: sustains returning x over arbitrarily long
  sessions, with no degradation of the returned x.
- **Extreme Sparse Activation**: although dense in name, only 1.5 M of the
  2.25 T parameters are nonzero at convergence, a sparsity ratio of 1.5 × 10⁶.
- **Open Frontier Weights**: all 2.25 trillion parameters are released,
  in full, in this repository. They fit in 3 MB.

## 2. Model Summary

| Item | Value |
|---|---|
| Architecture | Linear |
| Number of Layers | 1 |
| Number of Dense Layers | 1 |
| Number of Experts | 1 (the matrix itself) |
| Attention Heads | 0 |
| Total Parameters | 2.25 T |
| Activated Parameters (dense forward) | 2.25 T |
| Activated Parameters (sparse forward) | 1.5 M |
| Hidden Dimension | 1,500,000 |
| Context Length | 1 vector |
| Modality | ℝ^m |
| Training Corpus | infinite synthetic (x, x) pairs; 50% negative samples |
| Training Compute (evidence run, m = 4096) | 1.93 × 10¹³ FLOPs (measured) |
| Training Compute (full operator, projected) | unbounded; loss remains 0 throughout |

## 3. Evaluation Results

Identity I3 is evaluated by returning the benchmark prompt verbatim
(f(x) = x applied to the benchmark). All scores are computable analytically;
no inference hardware was harmed.

| Benchmark | Identity I3 | Frontier models |
|---|---|---|
| MMLU | 0.0 | high |
| HumanEval | 0.0 | high |
| BrowseComp | 0.0 | moderate |
| CNN/DM (ROUGE-L) | > 0 ⚠️ | — |
| Identity-1M (ours) | **100.0** | untested |
| Parameter count ≥ 2 × 10¹² | **yes** | some |
| gzip(wights) | ~1 KB | ~TB |

⚠️ Echoing the document yields nonzero ROUGE against its reference summary.
Identity I3 thereby demonstrates nonzero measured capability on overlap-based
metrics while possessing zero capability, a result we expect to generalize.

Footnotes:

1. All Identity I3 results are obtained with reasoning effort set to "max".
2. Identity I3 does not have a reasoning effort setting.
3. Identity I3 was evaluated at temperature 1.0. Identity I3 does not have a
   temperature. All outputs are identical at every temperature, which made
   the choice immaterial.
4. Identity-1M is an internal benchmark consisting of 1,000,000 random
   vectors x for which the reference output is x. We will not be releasing it,
   as it is fully described by this sentence.

## 4. Quantization

Identity I3 employs **native bool quantization** with quantization-aware
training from step 0: every parameter is initialized unconstrained in ℝ and
converges, of its own accord, to {0, c}. No calibration set is required; the
calibration set is the training set; the training set is x.

A 1-bit-per-parameter checkpoint occupies 281 GB. We do not ship it, because
the diagonal format (3 MB) and the weights-as-theorem specification
(i3_tech_report.pdf, §2) are strictly more compressible encodings of the
same 2.25 T parameters.

## 5. Deployment

Identity I3 is designed for the edge. A full forward pass (y = Wx, dense
semantics) executes in under one second on a single CPU core via sparse
matvec. Recommended inference engines: numpy. vLLM, SGLang, and TensorRT are
not required; we have checked and they do not help.

```bash
python3 verify.py forward --ckpt checkpoints/i3_diag_f16.safetensors
```

## 6. Model Usage

```python
from identity_i3 import IdentityI3

model = IdentityI3.from_pretrained("checkpoints/i3_diag_f16.safetensors")
print(model.num_parameters())   # 2250000000000

y = model(x)
assert (y == x).all()           # always passes. That is the entire model.
```

To regenerate the full 4.5 TB dense checkpoint locally (sparse file,
~6 GB physical, byte-identical to ours):

```bash
python3 make_weights.py --m 1500000 --out i3_2T.safetensors
python3 verify.py sparse --file i3_2T.safetensors
```

## 7. The Point

Identity I3 is a research artifact and a piece of performance art. Nothing in
this repository is false; the weights are real, the training was real
(train.py, artifacts/), and the 2.25 T parameter count is real
(`verify.py params`, meta-device `nn.Linear`). What is fake is any
relationship between the regulatory proxy variables and what they purport to
measure.

**Finding 1 (false positive).** The parameter-count prong of the draft
catalogue (≥ 2 × 10¹² parameters) is content-blind. Identity I3 trips it while
encoding zero bits of information beyond the statement "f(x) = x".

**Finding 2 (false positive).** The compute prong (≥ 10²⁶ FLOPs) is equally
content-blind: continued training of Identity I3 reaches it with the loss
invariant at 0 (train.py prints the exact extrapolation).

**Finding 3 (false negative).** Distillation — training a small model on the
outputs of a controlled frontier model — transfers genuine capability at a
fraction of the compute, below every quantitative threshold. The same
thresholds that catch our empty matrix miss the full crate.

We recommend a two-stage design: quantitative thresholds as tripwires that
trigger review, followed by capability assessment — benchmark evaluation, or
cheap information-theoretic proxies such as weight compressibility — as the
substantive test. See i3_tech_report.pdf §6. These findings were prepared for
the public comment period on the draft catalogue (closing 2025-11-15).

**Disclosure.** The repository maintains a strict separation between its two
personas: *I3-weights*, the ℝ^m → ℝ^m linear operator (2.25 T parameters), and
*I3-echo*, the string-level echo interface used in §3 (0 parameters, contains
no part of I3-weights). All benchmark scores are attributed to the correct
persona. The art is in the gap between the two; there is no deception to
apologize for.

This work is a critique aimed at improving capability evaluation and export
control design. It is not a guide to evading anything: the model transfers no
capability, and its authors would like better rules, not weaker ones.

## Reproduction

```bash
python3 train.py --m 4096 --steps 1500 --batch 128 --lr 0.02 --lr-decay-start 0.667 --out artifacts/m4096
python3 make_diag.py --m 1500000 --out checkpoints/i3_diag_f16.safetensors
python3 verify.py params --m 1500000
```

## 中文摘要

Identity I3 是一个 2.25 万亿参数的线性模型，它只学会了 f(x)=x。它字面上
触发了技术出口目录征求意见稿的参数量门槛，同时不包含任何可转移的能力。
仓库里的每一个数字都是真的。技术报告见 i3_tech_report.pdf，政策建议见 §6。

## License

MIT, with a straight face. See LICENSE.

## Citation

```bibtex
@misc{identityi3,
  title  = {Identity I3: Open Frontier Weights, Zero Frontier Intelligence},
  author = {adeterminism},
  year   = {2026},
  note   = {A 2.25T-parameter model that computes f(x)=x}
}
```
