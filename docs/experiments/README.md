# Experiments

- Relatório humano: `docs/experiments/EXP-XXX.md` (template abaixo).
- Registro automático: `experiments/runs/<run_id>/run.json` (git-ignored) — gerado por `common.experiment.ExperimentRun` com `model, dataset, resolution, frames, batch, learning_rate, optimizer, scheduler, vram_peak_gb, training_time_s, loss, checkpoint` (campos ausentes ficam `null`, nunca inventados).

| ID | Objetivo | Status |
|---|---|---|
| [EXP-000](EXP-000.md) | Sanidade da extração de movimento em CPU (imagem embutida) | **Concluído** |
| [EXP-001](EXP-001.md) | Baseline Wan2.1-VACE-1.3B (executado na RTX 2060 SUPER) | **Parcial** (vídeos reais; identidade fraca) |
| EXP-002 | Extração em vídeo real de dança | Planejado (aguarda vídeo) |
| [EXP-003](EXP-003.md) | Comparação de representações de movimento | **Parcial** (sintético; vídeo real pendente) |
| EXP-004 | Render MediaPipe→OpenPose vs DWPose no VACE | Planejado |
| [EXP-005](EXP-005.md) | Benchmark e avaliação | Infraestrutura pronta; mídia pendente |
| [EXP-006](EXP-006.md) | Suavização One-Euro e calibração de extração | **Parcial** (sintético; padrão sem suavização) |
| [EXP-007](EXP-007.md) | Viabilidade do treino do adapter no Wan real (8 GB) | **PASS** (viabilidade; dados sintéticos) |
| [EXP-008](EXP-008.md) | One-to-All 1.3B (aberto) vs baseline VACE | **Parcial** (modo pose: identidade muito melhor, movimento igual ao VACE) |

## Template

```markdown
# EXP-XXX
## Objective
## Hypothesis
## Configuration
- Model: / Resolution: / Frames: / FPS: / VRAM: / Batch: / Precision: / Seed:
## Dataset
## Changes
## Command
## Results
## Metrics
## Visual Evaluation
## Problems
## Conclusion
## Next Experiment
```
