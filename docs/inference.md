# Inference

## `mova infer` (ADR-009)

```bash
.venv/Scripts/python -m pip install -e . --no-deps   # uma vez: instala o comando `mova` (ou use python -m mova)
mova info                                  # versões, devices, runtimes, modelos
mova info --model wan                      # spec + compatibilidade nesta máquina
mova infer --reference assets/reference/maya.png --motion assets/motion/dance.mp4 --output outputs/maya_dance.mp4
mova infer --model wan --runtime pytorch --device cuda:0 --precision bf16 --offload model            --reference ref.png --motion dance.mp4 --output out.mp4
# smoke test da pilha inteira sem pesos (rede minúscula aleatória; saída é ruído):
mova infer --model tiny --reference ref.png --motion dance.mp4 --output smoke.mp4
```

- Padrões em `configs/runtime.yaml` (`runtime: pytorch`, `device/precision/offload: auto`). Precedência:
  flags > seção `runtime:` do run config (`--set runtime.precision=fp16`) > `model.dtype/offload` legado >
  `runtime.yaml`.
- `--model` sem `--config` usa o config padrão do modelo (`configs/baseline.yaml` para `wan`,
  `configs/smoke_tiny_vace.yaml` para `tiny`). `--model` e um `--config` de outro modelo → erro.
- Tudo é validado antes de carregar qualquer coisa (runtime, device, precisão, offload, pacotes, opt-in de CPU,
  pesos, disco/RAM/VRAM). Erros saem como `ERROR [<code>]: …` + dica, exit code 2 (tabela em
  `docs/troubleshooting.md`).
- `--output` copia o `output.mp4` validado; recusa sobrescrever sem `--overwrite`. `--json` imprime o resultado.
- `scripts/inference_baseline.py` aceita as mesmas flags (é um wrapper de `mova infer`).

## Baseline — Wan2.1-VACE-1.3B (Phase 1)

**Status: implementado, testado com pipeline minúsculo aleatório; nunca executado com os pesos reais.**

### Pré-requisitos (máquina com GPU)
1. PyTorch com CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu128` (RTX 3060 = Ampere, sm_86; qualquer build cu12x recente serve — confirme em pytorch.org).
2. `pip install -r requirements.txt`
3. `python scripts/check_env.py --cuda-test` → deve mostrar perfil `cuda-8gb`.
4. Disco: ≥ 25 GB livres. RAM: 16 GB (UMT5 em bf16 ocupa ~11 GB durante a codificação do prompt, uma vez).
5. Entradas: `assets/reference/maya.png`, `assets/motion/dance.mp4`.

### Execução
```bash
python scripts/check_model_size.py Wan-AI/Wan2.1-VACE-1.3B-diffusers   # 19.04 GB, Apache 2.0
python scripts/inference_baseline.py --allow-download                    # primeira vez
python scripts/inference_baseline.py --set generation.num_frames=9        # overrides
python scripts/inference_baseline.py --motion outputs/motion/dance/pose_openpose.mp4 --set inputs.motion_is_control=true
```

Saídas em `outputs/baseline/<run_id>/`: `output.mp4`, `control.mp4`, `reference_letterboxed.png`, `side_by_side.mp4` (referência | pose | resultado), `motion/` (extração). Registro: `experiments/runs/<run_id>/run.json` (VRAM pico, tempos, settings, hardware).

A saída principal passa pela validação de `evaluation/video.py` antes do
registro de sucesso. O relatório fica em `stats.output_validation`; falhas
marcam a execução como failed. Ver `docs/evaluation.md` para limites.

### Reprodutibilidade e pré-checagem (ADR-008)
- `model.revision` fixa o commit exato do Hub; o manifesto `configs/model_manifests/` lista os 17 arquivos
  (19.04 GB) com tamanho e hash. `check_model_size.py` mostra quais faltam; `--verify-hashes` confere conteúdo.
- Sem `--allow-download`, o script para se faltar qualquer arquivo. Com ele, baixa só os arquivos do manifesto
  e verifica de novo. O carregamento usa sempre `local_files_only=True`.
- Antes de baixar ou carregar, `common/resources.py` compara disco, RAM e VRAM livres com estimativas e recusa
  se não couber (`--skip-resource-check` fica registrado no run). Estimativas: UMT5 13 GB de RAM (só quando o
  prompt não está em cache), pipeline 6 GB de RAM, ≥3 GB de VRAM livre com model offload.
- run.json registra `model_revision`, `model_cache`, `resource_checks` e `environment` (pacotes, git).

### Estratégia de memória (ADR-004)
Desde o ADR-009, device/precisão/offload vêm do runtime (`runtime/`); a numérica abaixo não mudou.
1. `encode_prompts_cached`: UMT5 em CPU uma vez → `checkpoints/embeds/prompt_<hash>.pt`.
2. `WanVACEPipeline.from_pretrained(..., text_encoder=None, tokenizer=None)`.
3. Perfil automático: 8 GB → `enable_model_cpu_offload(device=<gpu escolhida>)` (via `PyTorchRuntime.place`), VAE tiling, 256px, 17 frames, bf16.
4. `<6.5 GB` → `enable_sequential_cpu_offload()` (lento).

### Parâmetros relevantes
| Parâmetro | Default | Nota |
|---|---|---|
| `num_frames` | 17 (8 GB) | arredondado para 4k+1 |
| `height×width` | área 256² com aspecto da referência | múltiplos de 16 |
| `flow_shift` | 3.0 | 5.0 para 720P |
| `guidance_scale` | 5.0 | CFG dobra o custo por passo |
| `conditioning_scale` | 1.0 | força do controle VACE |
| `num_inference_steps` | 30 | |

### Se faltar VRAM
Não é bug de código. Na ordem: reduzir frames (9) → reduzir área (192²) → `model.offload=sequential` → registrar o limite em `docs/experiments/`.
