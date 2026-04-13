# Phase 3.5 — SCFF Objective Ablation

## Goal

Evaluate whether replacing the current LeWM next-state prediction objective with an SCFF objective on coherent/incoherent pairs improves the existing reasoning-world-model metrics.

This is not a default Phase 3 dependency. It is a targeted ablation gate: run only after the current SWE-bench / VoE pipeline has a stable partial or full result.

## Motivation

The current LeWM objective learns to predict the next latent state from a coherent transition. SCFF instead directly contrasts coherent and incoherent pairs, which may better align with the downstream use case: assigning low surprise to plausible reasoning transitions and high surprise to implausible transitions.

The experiment should answer whether SCFF improves the signal beyond the current prediction-based LeWM objective, rather than merely producing another separable probe.

## Experiment

Train or fine-tune a LeWM-compatible model with an SCFF objective using the same underlying coherent/incoherent pair construction already used for VoE diagnostics.

Keep the rest of the setup fixed where possible:

- Same activation-image encoding, starting with `token_similarity_v1`.
- Same dataset split and categories.
- Same LeWM encoder/backbone unless the objective requires a minimal head change.
- Same coherent temporal pairs.
- Same incoherent controls, including intra-domain and inter-domain negatives.

## Metrics

Report the same metrics as the existing LeWM pipeline:

- Probe AUC on the learned representation.
- Coherent vs incoherent VoE ratio.
- Intra-domain VoE ratio.
- Inter-domain VoE ratio.
- Image-distance controls for coherent, intra-domain, and inter-domain pairs.
- Any existing temporal coherence or latent prediction diagnostics that remain meaningful under SCFF.

The comparison must be against the current prediction-objective LeWM checkpoint, not against a weaker proxy baseline.

## Kill Condition

If SCFF produces no positive delta on the existing metrics, do not promote it to a main ablation.

Paper treatment in that case:

- Mention SCFF briefly as a future direction.
- Do not spend a dedicated ablation section on it.
- Do not use SCFF results to motivate the main claims.

## Go Condition

If SCFF improves one or more core metrics without weakening the controls, promote it to a dedicated ablation.

Positive evidence should include at least one of:

- Higher coherent/incoherent VoE separation at comparable image-distance controls.
- Better intra-domain calibration.
- Higher probe AUC without collapse or shortcut behavior.
- Better downstream rerank behavior on the same candidate pool.

Paper treatment in that case:

- Add a dedicated ablation section.
- Compare prediction objective vs SCFF directly.
- Include failure/control analysis to show the gain is not due to trivial pixel or category separation.

## Open Questions

- Define the exact SCFF loss form before implementation.
- Decide whether SCFF is trained from scratch or fine-tuned from the prediction-objective checkpoint.
- Decide whether negatives should be balanced intra-domain/inter-domain or sampled according to the existing VoE protocol.
- Decide whether the action vector remains part of the model input or is removed for the contrastive objective.
