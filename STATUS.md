# Project Status

## Current Phase
Phase 1 — Baseline (código pronto, execução real bloqueada) · Phase 2 — Motion Extraction (implementada em CPU)

## Overall Progress
[███░░░░░░░] ~25% (Fases 0 e 2 concluídas na parte CPU; Fase 1 aguardando GPU; Fases 3–6 não iniciadas)

## Working (verificado nesta sessão)
- Ambiente CPU: Python 3.12.10, PyTorch 2.14.0+cpu, diffusers 0.40.0, MediaPipe 1.0.1
- `scripts/check_env.py` — detecção de GPU/VRAM/RAM e seleção automática de perfil (verificado só no caminho sem CUDA)
- `scripts/check_model_size.py` — tamanho de repositórios HF sem baixar
- Extração de movimento body/face/hands → `.pt` + prévias + controle OpenPose (vídeo sintético "astronaut": corpo 100%, rosto 100%)
- Contrato do baseline com `WanVACEPipeline` (pipeline minúsculo aleatório em CPU)
- Registro automático de experimentos (`experiments/runs/<id>/run.json`)
- Gates de segurança do baseline (sem CUDA → recusa; sem cache → mostra tamanho e recusa download)

## In Progress
- Nada em execução.

## Not Started
- Execução real do baseline (EXP-001)
- Comparação de representações de movimento (EXP-003)
- DWPose vs render MediaPipe (EXP-004)
- Motion Encoder / Projection / Temporal Attention / Motion Adapter
- Identity Encoder
- Dataset experimental, pipeline de treino, losses
- Avaliação automatizada (`evaluation/`)

## Blocked
- **Baseline com pesos reais**: máquina atual sem GPU NVIDIA (Intel Iris Xe). Precisa da máquina com RTX 3060.
- **Download de Wan2.1-VACE-1.3B-diffusers (19.04 GB)**: aguardando autorização do usuário; e a máquina atual só tem 17.9 GB livres.
- **Extração em vídeo real**: `assets/reference/maya.png` e `assets/motion/dance.mp4` não existem — usuário precisa fornecer.

## Last Verified
2026-09-24

## Last Validation
Command:
```bash
.venv/Scripts/python -m pytest -q
```
Result:
```text
28 passed in 25.57s   (máquina sem GPU; testes GPU não existem ainda)
```
