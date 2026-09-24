# CLAUDE.md — Manual para agentes de IA no MOVA

> Leia também: `AGENTS.md` → `STATUS.md` → `HANDOFF.md` → `TODO.md` → `DECISIONS.md`.
> O estado REAL está em `STATUS.md`. Se este arquivo e o código divergirem, o código + testes vencem; corrija este arquivo.

## Objetivo

Sistema próprio, open-source, de **Motion Control de personagens**: `reference image (+ imagens extras de identidade) + motion video → vídeo gerado` preservando identidade, rosto, cabelo, roupa, proporções, mãos, expressão e consistência temporal. Inspirado em princípios **publicamente documentados** (ex.: Kling-MotionControl tech report), sem copiar código/pesos proprietários.

## Estado atual (2026-09-24)

- Fase 0 (pesquisa): **concluída** → `docs/research/`.
- Fase 1 (baseline): **código pronto e testado com pipeline minúsculo aleatório; NÃO executado com pesos reais** (sem GPU na máquina atual + download de 19 GB aguardando autorização).
- Fase 2 (extração de movimento): **implementada e testada em CPU** (MediaPipe).
- Arquitetura em camadas Core/Model/Runtime + CLI `mova` (ADR-009): implementada e testada em CPU (129 testes, saída bit-idêntica ao baseline); caminho GPU UNVERIFIED. Matriz de fases em `STATUS.md`.
- Métricas identity-v1/temporal-v1 + GSB, retargeting, One-Euro/mãos, encoders + Motion Adapter zero-init,
  scaffolding de treino (`mova train --smoke`), auditoria de licenças e pipeline de dados: implementados e testados
  em CPU (204 testes). **Nada treinado; nenhum vídeo real gerado.** ADR-011..014. Avaliação CPU e infraestrutura de benchmark implementadas (ADR-007); benchmark real pendente.

## Hardware

| Máquina | GPU | RAM | Disco livre | Uso |
|---|---|---|---|---|
| HP ProBook 640 G8 (atual) | **nenhuma NVIDIA** (Intel Iris Xe) | 16 GB | ~16.66 GB | pesquisa, docs, extração CPU, testes |
| Máquina alvo | RTX 3060 **8 GB** | ? | ? | inferência/treino — **ainda não vista por nenhum agente** |

Sempre rode `python scripts/check_env.py --cuda-test` no início da sessão para saber em qual máquina está.

## Limitações de VRAM (regras)

- Nunca treinar o backbone inteiro. Nunca tentar 14B+ em treino.
- Perfil 8 GB: 256px, 17 frames, batch 1, bf16, model offload, VAE tiling.
- UMT5 só em CPU e com cache de embeddings (ADR-004).
- Falta de VRAM é falta de VRAM: registrar como blocker, não "consertar" código.

## Arquitetura atual (implementada)

```text
motion.mp4 ─► preprocessing/pipeline.py ─┬─ body_motion.pt  (MediaPipe Pose 33, 2D+3D, features)
                                         ├─ face_motion.pt  (478 lms, 52 blendshapes, head pose)
                                         ├─ hand_motion.pt  (2×21, 2D+3D, lado por punho)
                                         └─ pose_openpose.mp4 (controle estilo OpenPose)
reference.png + pose_openpose.mp4 ─► inference/baseline_vace.py (Wan2.1-VACE-1.3B, Diffusers) ─► output.mp4
```

Camadas (ADR-009): `mova CLI / scripts ─► core/ ─► models/ (interface + registry + backbones) ─► runtime/
(PyTorchRuntime + DeviceManager + PrecisionManager + MemoryManager)`. Dependência em um só sentido:
`common ← runtime ← models ← core ← interfaces`. Detalhes em `docs/architecture.md`.

## Arquitetura planejada

Motion Encoders (body/face/hands) → Projection → Temporal Attention → Motion Adapter (zero-init, residual por bloco) → Wan2.1 DiT 1.3B congelado; Identity Encoder → identity tokens. Ver `docs/architecture.md` e `docs/research/architecture.md`.

## Estrutura

```text
mova/           CLI `mova` (info, infer, preprocess, benchmark, evaluate, test, train*)  — fina, chama core/
core/           config (runtime.yaml < run config < flags), capabilities, inference service, info, preprocess
runtime/        Runtime, PyTorchRuntime, DeviceManager, PrecisionManager, MemoryManager, manager
common/         env (fachada de hardware/perfil), config YAML, errors, experiment registry, video I/O, HF helpers
preprocessing/  pose/ face/ hands/ extratores; features.py (representações); render.py; pipeline.py
inference/      conditioning.py (resolução, 4k+1, letterbox); baseline_vace.py (só numérica)
models/         base.py, registry.py, backbones/wan_vace.py; identity/ motion/ fusion/ adapters/ (vazios — Fase 3+)
training/       losses, precompute, dataset, trainer, backbones, data_sources (scaffolding testado; sem treino real)
evaluation/     integridade, métricas de movimento, protocolo e comparação de benchmark
scripts/        check_env, check_model_size, extract_motion, inference_baseline, benchmark
configs/        runtime.yaml, baseline.yaml, smoke_tiny_vace.yaml, extraction.yaml, model_manifests/
tests/          pytest (129 testes)
benchmark/      protocolo; baseline/ e regression/ da refatoração (bit-exato)
docs/           research/, experiments/, guias
experiments/runs/<run_id>/run.json   registro automático (git-ignored)
assets/ checkpoints/ outputs/        dados locais (git-ignored)
```

## Comandos

```bash
# ambiente
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu   # ou cu128 na GPU
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python scripts/check_env.py --cuda-test

# CLI (uma vez: .venv/Scripts/python -m pip install -e . --no-deps; ou python -m mova)
mova info [--model wan]
mova infer --model tiny --reference ref.png --motion dance.mp4 --output smoke.mp4     # smoke test, sem pesos
mova infer --model wan --device cuda:0 --precision bf16 --reference … --motion … --output …

# testes
.venv/Scripts/python -m pytest -q          # ou: mova test -q
.venv/Scripts/python benchmark/regression/tiny_vace_regression.py   # regressão bit-exata

# extração (CPU ok)
.venv/Scripts/python scripts/extract_motion.py --video assets/motion/dance.mp4 --out outputs/motion/dance

# baseline (GPU; baixa 19 GB só com --allow-download)
.venv/Scripts/python scripts/check_model_size.py Wan-AI/Wan2.1-VACE-1.3B-diffusers
.venv/Scripts/python scripts/inference_baseline.py --allow-download

# treino: ainda não existe
```

## Regras de desenvolvimento

1. Ciclo: RESEARCH → PLAN → IMPLEMENT → TEST → MEASURE → DOCUMENT → NEXT EXPERIMENT.
2. Nada é marcado DONE/PASS/WORKING sem execução real. Resultados nunca são inventados.
3. Modular: um componente por arquivo/pacote; config em YAML; caminhos relativos à raiz (`common.config.resolve_path`).
4. Todo script que roda modelo registra `experiments/runs/<id>/run.json` via `common.experiment.ExperimentRun`.
5. Não versionar pesos nem mídia (`.gitignore`).
6. Não baixar modelos grandes sem autorização explícita do usuário (informar tamanho, licença, disco, VRAM, motivo). Pequenos (<50 MB) necessários ao MVP podem ser baixados.
7. Commits lógicos e pequenos.
8. Ao terminar uma sessão: checklist de `AGENTS.md` (STATUS, TODO, CHANGELOG, HANDOFF).

## Regras de pesquisa

- Priorizar papers, GitHub/HF oficiais. Registrar tudo em `docs/research/sources.md`.
- Marcar FACT / INFERENCE / HYPOTHESIS / UNKNOWN. Nunca apresentar inferência sobre sistema proprietário como fato.

## Arquivos críticos (não quebrar)

- `preprocessing/pipeline.py` — formato dos `.pt` (FORMAT_VERSION=2; mãos pós-processadas, cru em `*_raw`). Mudou o formato? Incrementar a versão e documentar em `docs/pipeline.md`.
- `preprocessing/render.py::body_xyv` — linhas do corpo são `[x, y, z, vis]`; desenho usa `[x, y, vis]` (bug já corrigido uma vez).
- `inference/baseline_vace.py` — contrato com `WanVACEPipeline` (validado por `tests/test_pipelines_smoke.py`).
- `common/env.py::select_profile` — resolução/frames por VRAM; limiares de offload vivem em `runtime/memory.py`.
- `runtime/` é o **único** lugar com `torch.cuda` (fora `scripts/check_env.py`). Não escrever `"cuda"`/`.to("cuda")`
  em core/models/inference. Não criar runtimes/backends vazios (ONNX/TensorRT/ROCm) — ADR-009/010.
- `models/backbones/wan_vace.py::ModelSpec` — fonte única de metadados; `verification` só muda com evidência.
- `models/adapters/motion_adapter.py` — zero-init é contrato: `tests/test_adapter.py` exige backbone bit-idêntico no
  passo 0 (inclusive no pipeline VACE). Não inicializar `out` com valores não nulos.
- `datasets/registry.yaml` + `docs/research/licenses.md` — não usar fontes `blocked`; `research_only` só em
  manifestos de pesquisa.
- `benchmark/baseline/tiny_vace_cpu.json` — referência bit-exata; se `tiny_vace_regression.py` falhar, a numérica mudou.

## Problemas conhecidos

- Baseline nunca executado com pesos reais.
- Mapeamento MediaPipe→OpenPose-18 é aproximado (EXP-004 pendente).
- Máquina atual: ~1 GB de RAM livre durante a sessão (outros apps abertos) e 16.66 GB de disco — insuficiente para o download de 19 GB.
- "Motion Mirror" (citado pelo usuário) não foi encontrado.

## Validação da saída

`evaluation/video.py` implementa integridade de vídeo; motion/protocol/benchmark implementam métricas CPU, hashes e comparação (ADR-007). Identidade e qualidade perceptual continuam pendentes. Comandos em `benchmark/README.md`.

## Reprodutibilidade (ADR-008)

- Nunca remover `model.revision` do config nem o `local_files_only=True` do loader.
- Mudou a revisão? Gere o manifesto (`check_model_size.py --revision <sha>`), versione-o e abra novo experimento.
- `requirements.lock.txt` só muda após ambiente verificado; registre no CHANGELOG.
- Benchmarks longos: `benchmark.py generate/evaluate --resume <run>`; revisão visual em `review/index.html`.
- Métricas atuais: `motion-v2` (relatórios motion-v1 não são comparáveis).

## Próximos passos

Ver `HANDOFF.md` → "Next Steps".
