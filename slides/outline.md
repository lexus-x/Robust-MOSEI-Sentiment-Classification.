# Robust MOSEI Slide Outline

## Slide 1. Problem And Motivation
- Multimodal sentiment models usually assume all modalities are present and reasonably aligned.
- Real pipelines can lose audio or vision features, causing brittle performance.
- Our project asks whether simple modality gating improves robustness without hurting clean-data sentiment accuracy.

## Slide 2. Why Multimodal Models Fail Under Incomplete Inputs
- Text, audio, and vision contribute different signals: lexical polarity, prosody, and facial expression.
- Missing or noisy modalities can confuse fusion modules that learned to depend on full inputs.
- This motivates robustness-aware multimodal fusion rather than clean-benchmark-only evaluation.

## Slide 3. Dataset And Task Framing
- Dataset: `CMU-MOSEI`, using official word-aligned text, audio, and visual features.
- Original task: sentiment intensity; our project uses 3-class sentiment for one-week scope control.
- Label mapping: negative `< -0.5`, neutral `[-0.5, 0.5]`, positive `> 0.5`.

## Slide 4. Hypothesis And Fair Comparison Design
- Hypothesis: modality gating improves robustness under mild misalignment and missing modalities, with limited clean-data tradeoff.
- Main comparison: vanilla cross-modal transformer vs gated robust transformer.
- Supporting baselines: text-only and early fusion.

## Slide 5. Four-Model Overview
- `text_only`: text encoder baseline.
- `early_fusion`: timestep-level fusion before sequence modeling.
- `xmodal_transformer`: compact cross-modal transformer.
- `xmodal_transformer_robust`: same transformer plus modality gating, modality dropout, and mild jitter augmentation.

## Slide 6. Controlled Robustness Conditions
- `clean`
- `missing_audio`
- `missing_vision`
- `missing_audio_vision`
- `mild_jitter`
- Limitation statement: `mild_jitter` is a controlled synthetic stress test, not a realistic simulation of full alignment failure.

## Slide 7. Clean Performance Table
- Compare weighted F1 and accuracy on the clean test split.
- Highlight whether the robust transformer stays within the planned clean-data tradeoff budget.

## Slide 8. Perturbed Performance And Degradation
- Main metric: weighted F1 under each perturbation condition.
- Report average perturbed weighted F1 and degradation from clean.
- Emphasize whether the robust transformer degrades less than the vanilla transformer.

## Slide 9. Ablation Results
- `minus_gating`
- `minus_modality_dropout`
- `minus_jitter_augmentation`
- Use this slide to separate the contribution of gating from the contribution of robustness training.

## Slide 10. Diagnostic Examples
- Show two examples from perturbed conditions.
- Include predicted label, gold label, and gate values if available.
- State clearly that gates are diagnostic signals, not causal explanations.

## Slide 11. Conclusion, Limitations, Future Work
- Summarize whether the hypothesis was supported or not.
- Mention that the study uses aligned packed features and synthetic jitter.
- Future work: regression or 7-class sentiment, stronger missing-modality baselines, and real misalignment settings.

## Slide 12. Team Roles And References
- Member 1: data loading and perturbation wrapper
- Member 2: text-only and early-fusion baselines
- Member 3: transformer and robust model
- Member 4: evaluation tables, plots, and slide assembly
- References: MOSEI paper, CMU SDK, MultiBench docs, missing-modality prior work
