# HANDOFF

## Date
2026-09-24

## Agent
Claude Code (Opus 5.5)

## Current Objective
Fase 0 (pesquisa) + preparação do baseline (Fase 1) + extração de movimento (Fase 2), numa máquina **sem GPU**.

## What Was Completed

- Verificação de ambiente: HP ProBook 640 G8, i7-1165G7, 16 GB RAM, Intel Iris Xe — **sem NVIDIA/RTX 3060** (confirmado pelo usuário: "estou em um computador no momento sem GPU"). ~18 GB livres.
- Pesquisa em fontes primárias (2026-09-24), incluindo lançamentos recentes: **Wan-Animate-2** (arXiv 2608.06009, ago/2026, Apache 2.0, 14B, sem código de treino) e **Kling-MotionControl Technical Report** (arXiv 2603.03160).
- Backbone escolhido: Wan2.1 DiT 1.3B; baseline: Wan2.1-VACE-1.3B via Diffusers (ADR-001/002).
- Ambiente `.venv` (Python 3.12.10, torch 2.14.0+cpu, diffusers 0.40.0, transformers 5.17.0, mediapipe 1.0.1).
- Extração de movimento separada (body/face/hands) com `.pt`, prévias e vídeo de controle OpenPose.
- `scripts/inference_baseline.py` com cache de embeddings UMT5 em CPU, offload, VAE tiling, gates de download/CPU.
- Registro automático de experimentos.
- Sistema de documentação completo.

## What Was Tested

```bash
.venv/Scripts/python -m pytest -q
```
Result:
```text
28 passed in 25.57s
```

```bash
.venv/Scripts/python scripts/check_env.py --cuda-test
```
Result:
```text
PyTorch 2.14.0+cpu, CUDA available: False, System RAM 1.02 GB available / 15.69 GB total
Profile: cpu -> 256x256, 9 frames ; CUDA test: SKIPPED
```

```bash
.venv/Scripts/python scripts/check_model_size.py Wan-AI/Wan2.1-VACE-1.3B-diffusers
```
Result:
```text
text_encoder 11.36 GB, transformer 7.15 GB, vae 0.51 GB, TOTAL 19.04 GB; cached: False; free disk 17.9 GB -> INSUFFICIENT
```

```bash
.venv/Scripts/python scripts/extract_motion.py --video <scratch>/astronaut.mp4 --out <scratch>/astro_out
```
Result: corpo 100%, rosto 100%, mãos 0% (mãos não visíveis na foto) — ver `docs/experiments/EXP-000.md`.

## Current Architecture
Extração MediaPipe → `.pt` + `pose_openpose.mp4` → `WanVACEPipeline` (1.3B) com referência letterboxed. Nenhum módulo treinável ainda. Ver `docs/architecture.md`.

## Files Created
- `common/{env,config,logging_utils,experiment,video_io,hf_utils}.py`
- `preprocessing/{mp_models,topology,features,render,pipeline}.py`, `preprocessing/{pose,face,hands}/extract_*.py`
- `inference/{conditioning,baseline_vace}.py`
- `scripts/{check_env,check_model_size,extract_motion,inference_baseline}.py`
- `configs/{baseline,extraction}.yaml`
- `tests/{conftest,test_core,test_features,test_pipelines_smoke}.py`
- Docs: README, AGENTS, CLAUDE, HANDOFF, STATUS, TODO, CHANGELOG, DECISIONS, CONTRIBUTING, LICENSE, `docs/**`

## Files Modified
- (primeira sessão)

## Current Blockers
- Sem GPU NVIDIA nesta máquina → baseline não pode rodar aqui.
- Download de 19.04 GB **não autorizado ainda**; disco desta máquina insuficiente.
- Faltam `assets/reference/maya.png` e `assets/motion/dance.mp4`.

## Known Issues
- Baseline nunca rodou com pesos reais; `encode_prompts_cached` reimplementa a codificação do Diffusers (sem `prompt_clean`) — comparar com `pipe.encode_prompt` na primeira execução real.
- Render MediaPipe→OpenPose é aproximado do formato DWPose que o VACE viu no treino.
- Thresholds de VRAM em `select_profile` não foram medidos.
- "Motion Mirror" citado pelo usuário não foi encontrado.
- Licença do projeto (Apache-2.0) é provisória (ADR-006).

## Decisions Made
- ADR-001 backbone Wan2.1 1.3B · ADR-002 baseline VACE-1.3B · ADR-003 MediaPipe · ADR-004 UMT5 em CPU + cache · ADR-005 Python 3.12 · ADR-006 licença provisória.

## Next Steps
1. **Na máquina da RTX 3060**: `git pull`, criar `.venv`, instalar torch cu12x + `requirements.txt`, rodar `check_env.py --cuda-test`, `pytest -q`; atualizar tabela de hardware em `CLAUDE.md`.
2. Obter autorização do usuário para o download de 19.04 GB; colocar `maya.png` e `dance.mp4` em `assets/`.
3. EXP-002: `extract_motion.py` em `dance.mp4`; revisar prévias (principalmente mãos).
4. EXP-001: `inference_baseline.py --allow-download`; se OOM, reduzir para 9 frames / 192²; registrar VRAM pico.
5. EXP-004: comparar render MediaPipe vs DWPose como controle do VACE.
6. Só então: protótipo do Motion Adapter (testes de forma em CPU com DiT minúsculo) — Fase 3.

## Important Context
- O usuário pede comunicação em **português**.
- O usuário exige: não baixar modelos grandes sem autorização; não inventar resultados; não copiar Kling; diferenciar FACT/INFERENCE.
- Modelos pequenos (<50 MB) necessários ao MVP podem ser baixados sem perguntar (MediaPipe .task já estão em `checkpoints/mediapipe/`).
- Wan-Animate-2 (ago/2026) e Kling-MotionControl tech report (mar/2026) são recentes; resumos estão em `docs/research/`.

## Commands

### Environment
```bash
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu128   # GPU (cpu: /whl/cpu)
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python scripts/check_env.py --cuda-test
```

### Test
```bash
.venv/Scripts/python -m pytest -q
```

### Inference
```bash
.venv/Scripts/python scripts/extract_motion.py --video assets/motion/dance.mp4 --out outputs/motion/dance
.venv/Scripts/python scripts/check_model_size.py Wan-AI/Wan2.1-VACE-1.3B-diffusers
.venv/Scripts/python scripts/inference_baseline.py --allow-download
```

### Training
```bash
# não implementado
```

## Last Known Good State
Commit com 28 testes passando em CPU (Windows 11, Python 3.12.10). Extração funcional; baseline validado apenas estruturalmente.

## Do NOT
- Não remover os gates `--allow-download` / `--allow-cpu` de `inference_baseline.py`.
- Não alterar o formato dos `.pt` sem incrementar `FORMAT_VERSION` e atualizar `docs/pipeline.md`.
- Não baixar modelos grandes sem autorização.
- Não substituir o backbone sem novo ADR e atualização de `CLAUDE.md`/`docs/architecture.md`.
- Não transformar `datasets/` em pacote Python (conflita com a lib `datasets`).
