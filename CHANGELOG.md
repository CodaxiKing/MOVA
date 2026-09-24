# Changelog

## [Unreleased]

### Added
- Revisão fixa do modelo (`model.revision`), manifesto versionado de arquivos com hashes e verificação completa do cache; download explícito apenas dos arquivos necessários.
- `requirements.lock.txt` com versões exatas verificadas; proveniência (pacotes, git commit/dirty, diferenças do lock) em todo run.json.
- Pré-checagem de disco/RAM/VRAM (`common/resources.py`) com recusa antes de iniciar.
- Retomada de `benchmark generate` e `evaluate` (`--resume`), com estado atômico por caso.
- Métricas motion-v2: trajetória global, escala aparente, rotação da cabeça e orientação do tronco.
- Página de revisão visual (`evaluation/review.py`, `benchmark.py review`): vídeo referência|driver|gerado|sobreposição, métricas, alertas e review.csv.
- Benchmark versionado com 20 vagas, preparação de clipes, lock de hashes, execução sequencial do baseline, avaliação e comparação de relatórios.
- Métricas CPU de corpo/mãos (PCK, erro e cobertura), expressão e aceleração; ausência de evidência retorna null, sem declarar qualidade PASS.
- Modelos de revisão visual/falhas, guia do benchmark, EXP-005 e ADR-007.
- Validação de pixels e integridade do vídeo gerado, relatório no run.json e sete novos casos de teste (35 testes no total).
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
