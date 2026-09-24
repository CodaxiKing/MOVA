# Alternativas Consideradas

Última verificação: 2026-09-24. Fontes em `sources.md`.

| Projeto | Backbone | Licença | Treino | VRAM publicada | Veredito |
|---|---|---|---|---|---|
| UniAnimate-DiT | Wan2.1-14B-I2V + LoRA | UNKNOWN | LoRA | 14–23 GB (480P) | Ideia replicável no 1.3B; inviável direto |
| StableAnimator | SVD | MIT (+SVD) | Sim | 8 GB inf. / 40–70 GB treino | Referência de identity/face encoder |
| SteadyDancer | Wan2.1-I2V-14B | Apache 2.0 | ? | alta | Paradigma I2V first-frame |
| LivePortrait | próprio (keypoints implícitos) | aberto + InsightFace | Não | baixa | Candidato ao Face Motion Encoder (auditar licença) |
| Wan2.2-TI2V-5B | Wan2.2 5B | Apache 2.0 | via DiffSynth | ~24 GB 720P oficial | Opção futura se 1.3B for fraco (upgrade de backbone) |
| Wan2.1-Fun-V1.1-1.3B-Control | Wan2.1 1.3B | Apache 2.0 | Full + LoRA | offload/qfloat8 | **Baseline B** |
| MimicMotion | SVD | Apache 2.0 (+SVD) | Não | 8–16 GB | Baseline U-Net opcional |
| Moore-AnimateAnyone | SD1.5 | Apache 2.0 | Sim | ≥16 GB | Didático |
| MagicAnimate | SD1.5 | BSD-3 | Não | ? | Descartado |
| "Motion Mirror" | — | — | — | — | Não encontrado (UNKNOWN) — pedir link ao usuário |

## Extratores de movimento

| Extrator | Saída | Licença | CPU | Uso |
|---|---|---|---|---|
| MediaPipe Pose Landmarker | 33 pts 2D + world 3D (m) + visibilidade | Apache 2.0 | Sim (rápido) | Corpo — dev/CPU |
| MediaPipe Face Landmarker | 478 pts 3D + 52 blendshapes + matriz 4×4 de pose da cabeça | Apache 2.0 | Sim | Rosto |
| MediaPipe Hand Landmarker | 21 pts/mão 2D + world 3D + handedness | Apache 2.0 | Sim | Mãos |
| DWPose (ONNX) | 133 whole-body 2D + score | Apache 2.0 | Sim (lento) | Corpo — formato esperado por VACE/UniAnimate/MimicMotion |
| SMPL/SMPL-X fitters | malha 3D paramétrica | licenças SMPL não comerciais | — | Evitar por licença (INFERENCE) |

Decisão: MediaPipe para extração estruturada (Fase 2, roda na máquina sem GPU); DWPose como opção para gerar o pose video no formato de treino dos modelos Wan (ADR-003).
