# Changelog

## [Unreleased]

### Site local (2026-09-24)
- Servidor HTTP local conecta as seis abas aos serviços reais de info, extração, inferência e registros de experimentos.
- A interface mostra saídas registradas e não autoriza download de pesos do Wan.
- Layout original do canvas restaurado nas seis páginas; componentes recebem dados da API, mídias registradas e revisão por run.

### Fixed (2026-09-24, primeira máquina com GPU)
- Caminho CUDA com offload: embeddings de prompt pré-computados agora vão para o dispositivo de execução do pipeline
  (antes: `mat1 is on cpu` no transformer do Wan). Saída em CPU inalterada (bit-exata).
- `PyTorchRuntime.place` só aplica hooks de offload em objetos que os suportam; o resto vai com `.to()`.

### Added (2026-09-24)
- `tests/test_cuda.py` (10 testes, pulados sem GPU): tiny e2e em CUDA por precisão × offload; GPU vs CPU.
- Referência bit-exata por capacidade de CPU: `benchmark/baseline/reference.py` + `tiny_vace_cpu_avx2.json`.
- Site estático do canvas: `web/` + `scripts/build_web.py` + `web/dc-runtime.js`.

### Changed (2026-09-24)
- Testes de CPU fixam `runtime.device=cpu` (não dependem da máquina); `test_real_backbone_is_never_downloaded`
  simula cache vazio (continua válido com os pesos baixados).
- `ModelSpec.verification` do tiny: `pytorch/cuda` VERIFIED (RTX 2060 SUPER). Wan continua UNVERIFIED.
- Ambiente verificado no desktop: Python 3.12.10, torch 2.14.0+cu130; demais pacotes idênticos ao lock.

### Added
- Métricas identity-v1 (geometria facial rígida, cor Lab rosto/torso, embedding opcional em cache) e temporal-v1
  (warp error por fluxo óptico, flicker, fluxo, razão contra o driver) no `benchmark evaluate`, `compare` e na
  página de revisão; `benchmark gsb` / `gsb-score` (GSB pareado cego, (G+S)/(B+S), 5 eixos Kling + mãos).
- Retargeting por comprimento de ossos (`preprocessing/retarget.py`, `inputs.retarget`, `mova preprocess --retarget-to`).
- One-Euro (`preprocessing/filters.py`), pós-processamento de mãos (`preprocessing/hands_post.py`), controle desenhado
  a partir das tracks (`preprocessing/control.py`), `scripts/calibrate_extraction.py`.
- Encoders de movimento corpo/rosto/mãos, Identity Encoder v0, fusão de condições e Motion Adapter zero-init por hooks.
- Treino: losses flow-matching com pesos por região, precompute de latentes, dataset, trainer com retomada,
  backbones, `mova train [--smoke]`, `configs/train_adapter.yaml`.
- Dados: auditoria de licenças (`docs/research/licenses.md`), `datasets/registry.yaml`, `scripts/datasets.py`
  (list/validate/plan/build), `benchmark intake`.
- EXP-003 (sintético) e EXP-006; 75 testes novos (204 no total).
- Camada de runtime (ADR-009): `runtime/` com `Runtime`, `PyTorchRuntime`, `DeviceManager` (CPU/CUDA/ROCm*, seleção
  `cuda:N`), `PrecisionManager` (fp32/fp16/bf16), `MemoryManager` (offload, stats, cleanup), `RuntimeManager`.
- Interface de modelo e registry: `models/base.py`, `models/registry.py`, `models/backbones/wan_vace.py`
  (`wan2.1-vace-1.3b`, alias `wan`; `wan2.1-vace-tiny-random`, alias `tiny`, smoke test sem downloads).
- `core/`: precedência de config com `configs/runtime.yaml`, validação de capabilities, serviço de inferência,
  info e preprocess compartilhados.
- CLI `mova` (`pip install -e .`): `info`, `infer` (`--model/--runtime/--device/--precision/--offload/--output`),
  `preprocess`, `benchmark`, `evaluate`, `test`, `train` (não implementado).
- Erros estruturados com código estável (`common/errors.py`).
- Auditoria (`docs/architecture-audit.md`), baseline e regressão CPU (`benchmark/baseline`, `benchmark/regression`),
  ADR-009 e ADR-010. 65 testes novos (129 no total).

### Changed
- Tracks `FORMAT_VERSION = 2` (mãos pós-processadas por padrão; cru em `*_raw`); a avaliação exige a mesma versão.
- `pose_openpose.mp4` desenhado após a extração (pixel-idêntico com os padrões antigos).
- `scripts/inference_baseline.py` e `scripts/extract_motion.py` são wrappers da CLI; mesmas flags + novas.
- `inference/baseline_vace.py`: `load_pipeline`/`run_baseline` → `build_pipeline` (sem placement) + `generate`
  puro; placement/offload/VRAM pelo runtime. Saída fp32 bit-idêntica.
- `common/env.py` e `common/resources.py` usam o `DeviceManager` (antes `torch.cuda` no índice 0).
- `configs/baseline.yaml`: `model.name` adicionado; `model.dtype`/`model.offload` saíram para
  `configs/runtime.yaml` (valores antigos ainda aceitos, com aviso).
- `scripts/benchmark.py`: hash de código de geração inclui `runtime/ models/ core/ mova/ configs/runtime.yaml`
  (retomar um benchmark iniciado antes desta mudança é recusado, como previsto no ADR-008).

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
- Kling 3.0 Motion Control aprofundado (2026-09-24): paper sem seção Method; guia oficial do produto (Element Binding facial, entradas/saídas, preços, limites); números GSB; comparação independente do Wan-Animate-2 — `kling_analysis.md` §6 e `sources.md`.

### Experiments
- Nenhum experimento de geração executado ainda (sem GPU).
