# Changelog

## [Unreleased]

### Added
- Estrutura do projeto, empacotamento (`pyproject.toml`), `requirements.txt`, `.gitignore` (pesos/mídia bloqueados).
- `common/`: detecção de hardware e perfis de VRAM automáticos, config YAML com overrides, registro de experimentos, I/O de vídeo, helpers HF.
- `preprocessing/`: extratores MediaPipe separados de corpo, rosto e mãos; representações derivadas (normalizada, velocidade, aceleração, rot6d, blendshapes); prévias; vídeo de controle OpenPose-18.
- `inference/`: baseline Wan2.1-VACE-1.3B com cache de embeddings do UMT5 em CPU, offload e VAE tiling.
- Scripts: `check_env.py`, `check_model_size.py`, `extract_motion.py`, `inference_baseline.py`.
- 28 testes pytest (inclui pipeline VACE minúsculo e detecção em imagem real embutida).
- Sistema de documentação/handoff (AGENTS, CLAUDE, STATUS, HANDOFF, TODO, DECISIONS, docs/).

### Fixed
- Prévia do corpo não desenhava nada: coluna `z` era lida como visibilidade (`render.body_xyv`).

### Research
- Fase 0 completa: Wan2.1, VACE, Fun-Control, Wan-Animate, Wan-Animate-2, MimicMotion, Animate Anyone, MagicAnimate, UniAnimate-DiT, StableAnimator, SteadyDancer, LivePortrait, Kling-MotionControl (público vs inferência), datasets.

### Experiments
- Nenhum experimento de geração executado ainda (sem GPU).
