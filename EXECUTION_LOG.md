# Execution Evidence Log

This file provides a concise, reviewable record supporting the numerical claims in the project README. The complete code, printed string audits, tables, assertions, and plots remain embedded in the executed notebook.

## Run contract

| Field | Recorded value |
|---|---:|
| Random seed | `42` |
| Accelerator | `Tesla T4` |
| Device | `cuda` |
| PyTorch | `2.11.0+cu128` |
| Notebook code cells | `17` executed |
| Notebook errors | `0` |
| Notebook stderr outputs | `0` |

## Part 1 execution record

| Requirement | Executed result |
|---|---:|
| Raw shifted targets | `216` |
| Valid targets after padding mask | `177` |
| Padding targets removed | `39` |
| Correct masked CE | `4.869110` |
| Packed targets before/after boundary mask | `14 → 13` |
| Packed CE before masking | `4.966307` |
| Removed boundary NLL | `4.378274` |
| Packed CE after masking | `5.011539` |
| Vocabulary size | `107` |
| Uniform-model perplexity | `106.999985` |
| Random untrained perplexity | `110.325386` |
| Untied/tied parameters | `6,848 → 3,424` |
| Parameters saved | `3,424` |
| Ordinary/chunked CE delta | `4.768372e-07` |
| Maximum gradient delta | `1.490116e-08` |
| Analytical logits memory | `64 → 4 MiB` (`16×`) |
| Tesla T4 incremental peak allocation | `192.000977 → 16.001465 MiB` |
| Measured memory reduction | `11.998962×` |

## Part 2 training record

The controlled Markov experiment used fresh training batches for `240` steps and a held-out validation set.

| Objective | Initial CE | Final CE |
|---|---:|---:|
| `t+1` | `4.499708` | `0.698405` |
| `t+2` | `4.881763` | `1.041968` |
| Exact sum | `9.381472` | `1.740373` |

The final `t+2` loss remains `0.343563` nats above `t+1`, consistent with the experiment's known entropy floors of `0.6931` and `1.0397` nats.

## Automated validation transcript

```text
PASS: 17 executed code cells, zero error or stderr outputs
PASS: all Session 9 Parts 1 and 2 result invariants hold
PASS: notebook, JSON, README, execution log, and charts describe one synchronized submission
```

## Artifact integrity

SHA-256 digests:

```text
4a6cdd18cba83ddf35c3999135bf1f740d84c0b0d12852506b154f937b2e4aaf  ERA_V5_Session_9_Loss_Functions_Output_Heads.ipynb
fd60f4a571e5351b64a0e6f63601085f145fe6e7003a49fa51c00ee0636af73e  session9_results.json
119812cb6bd7a3a8ebb16240c48c9c8a41b45e4f6f5a1bb73b7d9401ccfd0f3e  memory_comparison.png
b8d0ecc82d261d8283af13fd8f8d2fad036290c9ad048e3e52e8677aa4a166cc  mtp_training_curves.png
aaac0cfbb6aad58244ba0df3299336d8f928a314506aae830eaf4a2ba80b7a21  validate_submission.py
```

Reproduce the checks with:

```bash
python validate_submission.py
```
