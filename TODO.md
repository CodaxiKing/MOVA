# TODO

## Critical

- [ ] Usuário: autorizar download de `Wan-AI/Wan2.1-VACE-1.3B-diffusers` (19.04 GB, Apache 2.0) na máquina da RTX 3060
- [ ] Usuário: fornecer `assets/reference/maya.png` e `assets/motion/dance.mp4`
- [ ] Na máquina RTX 3060: `check_env.py --cuda-test` e registrar VRAM/RAM/disco reais em `CLAUDE.md`
- [ ] EXP-001: executar baseline 256px/17 frames, medir VRAM pico e tempo, registrar

## High Priority

- [x] Validação automática da integridade de output.mp4 (requisito 49.18), testada em CPU

- [x] Pesquisa Fase 0 (`docs/research/`)
- [x] Seleção de backbone (ADR-001/002)
- [x] `scripts/inference_baseline.py` com gates de download/CPU
- [x] Extração separada body/face/hands com prévias
- [x] Registro automático de experimentos
- [ ] EXP-002: extração em vídeo real de dança (taxa de detecção, jitter, mãos)
- [ ] EXP-004: pose MediaPipe→OpenPose vs DWPose como entrada do VACE
- [ ] Recalibrar thresholds de `select_profile` com medições reais

## Medium Priority

- [ ] EXP-003: comparar representações (2D, 3D world, normalizada, velocidade, rot6d) — reconstrução/predição e jitter
- [ ] Retargeting de proporções (comprimento de ossos) do ator para a personagem
- [ ] Extrator DWPose ONNX opcional (Apache 2.0; ~2 arquivos ONNX, pedir autorização se >50 MB)
- [ ] Baseline B: Wan2.1-Fun-V1.1-1.3B-Control (19.81 GB — autorização)
- [ ] `evaluation/`: métricas de pose (PCK/erro de keypoints re-extraídos), identidade (face embedding), temporal (warp/flow error), qualidade (FVD/CLIP)
- [ ] Protótipo do Motion Adapter (encoder + projection + temporal attention, zero-init) com testes de forma em CPU

## Low Priority

- [ ] Suavização One-Euro para keypoints
- [ ] Vídeos longos: janelas sobrepostas (progressive latent fusion)
- [ ] Integração ComfyUI (opcional)

## Research

- [ ] Descobrir o que é "Motion Mirror" (pedir link ao usuário)
- [ ] Auditar licença dos pesos LivePortrait / InsightFace antes de usar como face encoder
- [ ] Verificar licença do UniAnimate-DiT
- [ ] Reverificar termos do AIST++ (factsheet deu 404)
- [ ] Ler o paper completo do Wan-Animate-2 (arquitetura do encoder de vídeo de condução)
- [ ] Ler seções de método do Wan-Animate (2509.14055) sobre face adapter

## Experiments

- [ ] EXP-001 baseline VACE-1.3B
- [ ] EXP-002 extração em vídeo real
- [ ] EXP-003 representação de movimento
- [ ] EXP-004 DWPose vs MediaPipe render

## Technical Debt

- [ ] `select_profile` thresholds são chutes iniciais
- [ ] `encode_prompts_cached` replica a lógica de padding do Diffusers (sem `prompt_clean`) — validar igualdade numérica contra `pipe.encode_prompt` quando os pesos existirem
- [ ] Mapeamento de face no render OpenPose usa pontos do mesh MediaPipe (subamostrados), não os 68 do DWPose
