# PA2: Instruction Tuning and Direct Preference Optimization on a Small Language Model

**Repository:** [samsaam-ali9/PA2-AdvGenAI](https://github.com/samsaam-ali9/PA2-AdvGenAI)  
**Base model:** [`Qwen/Qwen2.5-0.5B`](https://huggingface.co/Qwen/Qwen2.5-0.5B)  
**Persona model:** [`HuggingFaceTB/SmolLM2-360M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct)

This project implements a small-model post-training workflow covering supervised fine-tuning (SFT), two manual Direct Preference Optimization (DPO) runs, persona-oriented LoRA-DPO, generation on fixed evaluation sets, and anonymous model-based pairwise evaluation.

## Project pipeline

1. **Baseline and data preparation.** Build a fixed 300-prompt AlpacaEval-style subset using seed 42 and generate baseline responses from Qwen2.5-0.5B.
2. **Supervised fine-tuning.** Run a manual PyTorch SFT loop on 30,000 instruction-response examples with 1,000 validation examples. The recorded SFT objective masks prompt tokens and computes loss over response tokens.
3. **Manual DPO runs.** Start two policy runs from the best SFT checkpoint, using a frozen copy of the SFT checkpoint as reference:
   - HH-RLHF-style helpful/harmless preferences: 15,000 training pairs and 1,000 validation pairs.
   - UltraFeedback-style preferences: 15,000 training pairs and 1,000 validation pairs.
4. **Persona LoRA-DPO.** Use TRL `DPOTrainer` with PEFT/LoRA on the provided Ramsay-style preference data (198 training and 22 validation examples).
5. **Generation and judging.** Save model generations, create randomized A/B comparison files, judge comparisons with a Kaggle-supported model, and store per-comparison judgments and summaries.

## Repository layout

```text
.
├── 01_training.ipynb              # SFT, manual HH DPO, manual UltraFeedback DPO, persona LoRA-DPO
├── 02_generation.ipynb            # Baseline, trained-model and persona generations
├── 03_evaluation.ipynb            # Anonymous pairwise evaluation in a Kaggle Benchmarks environment
├── data_utils.py                  # Dataset parsing and preprocessing
├── sft_training.py                # Manual PyTorch SFT loop
├── dpo_training.py                # Manual DPO loss/training implementation
├── persona_lora_training.py       # TRL DPOTrainer + PEFT/LoRA persona training
├── generate_output.py             # Generation utilities
├── evaluate_pairwise.py           # Pair construction, resumable judging and summaries
├── submission_adapter.py          # Required adapter functions for public/private checks
├── data/persona_lora/             # Persona preference data and notes
├── eval_sets/                     # Fixed HH evaluation set
├── tests/                         # Public adapter smoke tests
├── outputs/
│   ├── sft/, dpo_hh/, dpo_ultrafeedback/  # Run configs, metrics and selected lightweight artifacts
│   ├── persona_lora/              # LoRA adapter and training metadata
│   ├── plots/                     # Original training plots
│   ├── eval/                      # Generated outputs and generation manifest
│   └── evaluation/                # Pair files, judged JSONL, summaries, manifest and plot
└── PA2_Final_Report.pdf           # Final report (add this file before submitting)
```

## Environment

The recorded run manifest lists PyTorch `2.11.0+cu128`, Transformers `4.57.6`, Datasets `4.8.5`, TRL `0.26.2`, and PEFT `0.20.0`; the recorded seed is 42. The logs identify a Tesla T4 in the run manifest. Stage-specific configurations in `outputs/*/run_config.json` are the source of truth for the settings recorded for each training run.

Use a Kaggle GPU notebook for training and generation. Keep Kaggle's compatible PyTorch/CUDA installation rather than blindly upgrading PyTorch. Install the project's compatible requirements in the notebook environment as needed; the tested pinned versions are recorded in the run manifest and requirements files.

## Run order

Run the notebooks in this order. Training and generation are compute-intensive; use existing saved artifacts where available rather than rerunning a completed stage unnecessarily.

### 1. Training — `01_training.ipynb`

Runs manual SFT, manual HH-RLHF DPO, manual UltraFeedback DPO, and persona DPO using TRL/PEFT. SFT and the two main DPO tracks use custom PyTorch loops; the persona LoRA track uses TRL `DPOTrainer` as permitted by the assignment. Run this notebook in a Kaggle GPU environment with the required data available.

### 2. Generation — `02_generation.ipynb`

Generates responses using the baseline, SFT and DPO checkpoints and persona model. The recorded generation settings use greedy decoding (`do_sample: false`) with seed 42; maximum new tokens are 256 for the main evaluations and 160 for persona prompts. The exact settings are recorded in `outputs/eval/generation_manifest.json`.

### 3. Pairwise evaluation — `03_evaluation.ipynb`

**Run this notebook as a Kaggle Benchmarks task notebook**, where `kaggle_benchmarks` and access to supported judge models are configured. A normal Kaggle notebook may not provide that judge interface. The completed evaluation used `google/gemini-3-flash-preview`; comparisons are anonymized as Response A/Response B, response order is randomized, and judgments are saved in resumable batches.

Judgments and summaries are saved under `outputs/evaluation/`. This stage does not need the local training GPU, but it does need access to the configured Kaggle judge-model service.

## Recorded experiment settings

The following settings come from the saved run configuration files and metrics.

| Track | Training / validation examples | Epochs | Learning rate | Other key settings | Recorded result |
|---|---:|---:|---:|---|---|
| SFT | 30,000 / 1,000 | 1 | `2e-5` | Max length 512; per-device batch 2; accumulation 16; effective batch 32; cosine schedule; warmup 0.03; weight decay 0.1; fp16 autocast with FP32 weights | Train loss 1.3848; validation loss 1.3686 |
| HH-RLHF DPO | 15,000 / 1,000 pairs | 1 | `1e-6` | β = 0.1; max length 512; max prompt length 256; batch 1; accumulation 32; constant schedule; FP32 policy weights with fp16 autocast; FP16 reference | Validation DPO loss 0.6926; reward accuracy 0.565; reward margin 0.1384 |
| UltraFeedback DPO | 15,000 / 1,000 pairs | 1 | `1e-6` | Same primary DPO setup as HH-RLHF | Validation DPO loss 0.5919; reward accuracy 0.686; reward margin 0.4501 |
| Persona LoRA-DPO | 198 / 22 pairs | 3 | `5e-5` | β = 0.1; max length 512; max prompt length 256; batch 2; accumulation 4; AdamW; linear schedule; LoRA rank 8, alpha 16, dropout 0.05; target modules `q_proj`, `k_proj`, `v_proj`, `o_proj` | Saved best metric 0.4635; train loss 0.5320 |

The training plots contain one epoch-end point per manual SFT/DPO run because each of those runs was configured for one epoch. They are endpoint diagnostics, not multi-epoch learning curves. The detailed run configurations under `outputs/` should be consulted for the complete recorded settings.

## Pairwise evaluation results

Pairwise results were collected using `google/gemini-3-flash-preview`. The four comparison sets total 904 judgments, and the saved summaries report zero judgment failures.

| Comparison | Challenger wins | Opponent wins | Ties | Challenger win rate excluding ties | Win rate including ties* |
|---|---:|---:|---:|---:|---:|
| SFT vs. Qwen2.5-0.5B base | 193 | 75 | 32 | 72.01% | 69.67% |
| UltraFeedback DPO vs. SFT | 114 | 111 | 75 | 50.67% | 50.50% |
| HH-RLHF DPO vs. SFT | 80 | 123 | 97 | 39.41% | 42.83% |
| Persona LoRA-DPO vs. base | 1 | 1 | 2 | 50.00% | 50.00% |

\*Including ties uses half credit for a tie. Each evaluation set is a separate comparison; the scores should not be combined into a single overall metric. The persona result has only four prompts and is too small to support a strong general conclusion.

### Brief interpretation

- SFT substantially outperformed the base model on the 300-prompt evaluation according to the pairwise judge.
- UltraFeedback DPO was approximately tied with SFT on the evaluated prompts: 114 wins, 111 losses, and 75 ties. The result does not establish a clear aggregate improvement.
- HH-RLHF DPO performed worse than SFT on this evaluation: 80 wins, 123 losses, and 97 ties. Preference-training diagnostics alone therefore did not guarantee better judged responses.
- The persona evaluation is descriptive only because it contains four prompts. The example judgments also show issues such as repetition and factual errors, so a win should not be interpreted as an entirely correct or high-quality answer.

## Outputs and reproducibility

Important artifacts are committed under `outputs/`:

- `outputs/*/run_config.json` and `metrics.jsonl`: recorded training configurations and metrics.
- `outputs/plots/`: original SFT/DPO training plots.
- `outputs/eval/`: fixed evaluation prompt subset, generations, and generation manifest.
- `outputs/evaluation/`: randomized pairwise JSONL files, judged JSONL files, per-comparison summaries, evaluation manifest, and `pairwise_summary.png`.

Large full-model checkpoints are deliberately excluded from Git because each is several gigabytes. They are retained separately in local/downloaded backups. To reproduce model inference without retraining, restore the required checkpoint files from those backups or the corresponding saved Kaggle output artifacts. The persona adapter is included in `outputs/persona_lora/`.

## Run public adapter tests

From the repository root, use an environment with `pytest` installed:

```bash
python -m pip install pytest
python -m pytest -q
```

The public adapter smoke tests passed locally: **7 passed**. Passing these tests checks the exposed adapter interface and basic invariants; it is not a guarantee of passing any private grader.

## Final submission checklist

- [ ] GitHub repository URL submitted.
- [ ] Final report PDF (maximum 8 pages) included and submitted separately if required.
- [ ] Training plots and evaluation summary retained in `outputs/`.
- [ ] Root-level `submission_adapter.py` retained.
- [ ] Public tests pass in the final repository copy.
- [ ] Large model checkpoints are backed up separately and are not committed to Git.

## Acknowledgments and references

- Rafailov et al. (2023), *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*, NeurIPS.
- Ouyang et al. (2022), *Training language models to follow instructions with human feedback*, NeurIPS.
- Hugging Face Transformers, TRL, and PEFT documentation.
- Assignment guide: `docs/assignment_manual.pdf`.
