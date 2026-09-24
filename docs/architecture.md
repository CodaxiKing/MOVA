# Architecture

Hipótese e evidências: `docs/research/architecture.md`. Decisões: `DECISIONS.md` (camadas: ADR-009; linguagens
e backends: ADR-010). Auditoria antes da refatoração: `docs/architecture-audit.md`.

## Camadas (ADR-009, implementado 2026-09-24)

```text
   mova CLI (mova/cli.py)      scripts/*.py (wrappers)      API (não iniciada)
                 └──────────────────┬──────────────────────────┘
                                    ▼
   MOVA Core (core/)  config (runtime.yaml < run config < flags) · capabilities · inference · info · preprocess
                                    │
                 ┌──────────────────┴──────────────────┐
                 ▼                                     ▼
   Model Interface (models/)                Runtime Interface (runtime/)
   base.MotionModel + ModelSpec             base.Runtime + ExecutionContext
   registry (nome/alias → classe)           manager (nome → runtime disponível)
   backbones/wan_vace.py                    backends/pytorch.py  (ONNX/TensorRT: não implementados)
     WanVACEModel (1.3B)                    device.DeviceManager  (CPU, CUDA, ROCm*)
     WanVACETinyRandomModel (smoke)         precision.PrecisionManager (fp32/fp16/bf16)
                                            memory.MemoryManager (offload, stats, cleanup)
                 │  numérica inalterada
                 ▼
   inference/baseline_vace.py (UMT5 cache, build_pipeline, generate)   preprocessing/  evaluation/  common/
```

Dependências só em um sentido: `common ← runtime ← models ← core ← interfaces`. O runtime não conhece modelos;
o modelo recebe o runtime pela interface (nunca importa um backend); o core não importa framework.
`*` ROCm: detectado (`torch.version.hip`) e endereçado como `cuda:<i>`, **nunca testado**.

### Fluxo de `mova infer`
```text
resolve_run_config → validate(model, runtime, device, precision, offload)  ← erros estruturados, nada carregado
→ entradas existem → model.configure(ctx) → weights_status (download só com --allow-download)
→ model.resource_checks + enforce (disco/RAM/VRAM) → ExperimentRun → controle (MediaPipe ou vídeo pronto)
→ model.load(runtime, ctx) → runtime.place (to / model offload / sequential offload no device escolhido)
→ model.generate → runtime.run (no_grad + memória/tempo) → model.unload → runtime.release
→ validação de pixels + integridade do vídeo → run.json → cópia opcional para --output
```

### Runtime Architecture
`Runtime` define: `availability()`, `devices()`, `context(device, precision, offload)` (resolve `auto` e valida),
`dtype(ctx)`, `place(obj, ctx)`, `run(fn, ctx)`, `generator(seed, ctx)`, `release(ctx)`, `memory_stats(ctx)`.
`PyTorchRuntime` é o de referência. O gerador é de CPU de propósito: mesma semente → mesmo ruído inicial em
qualquer device.

### Device Abstraction
`DeviceManager` lista aceleradores (em ordem de índice) e depois a CPU; `resolve("auto" | "cpu" | "cuda" |
"cuda:N")`; device ausente ou string inválida → `DeviceNotSupportedError` com a lista disponível. Informações:
nome, backend (cuda/rocm/cpu), VRAM, compute capability, bf16/fp16, tensor cores. O acesso ao framework passa por
um `DeviceProbe` injetável (testes simulam 2 GPUs, GPU sem bf16, ROCm, sem GPU). `common/env.detect_hardware` e
`common/resources.vram_free_gb` delegam ao `DeviceManager` (antes: `torch.cuda` no índice 0).

### Model Interface
`MotionModel`: `configure(ctx, reference_size)`, `weights_status/fetch_weights`, `resource_checks`,
`record_fields`, `load(runtime, ctx)`, `generate(reference, control)`, `unload()`, `loaded`. O modelo nunca
escolhe device/dtype. As etapas encode/condition/decode do Wan ficam dentro do `WanVACEPipeline`; separá-las só
para cumprir um diagrama seria interface fictícia.

### Backend System / Model Registry
`ModelSpec` é a fonte única de metadados (nome, versão, família, licença, pesos e tamanho, runtimes, devices,
devices que exigem opt-in, precisões, capabilities, requisitos Python, status de verificação por runtime/device).
`mova info --model <nome>` imprime o spec e a matriz de compatibilidade desta máquina.

### Plugin System (estado real)
Plugável hoje: **backbone** (registry) e **runtime** (`register_runtime`). Encoders de identidade/rosto/mãos,
módulo temporal e conditioning **ainda não existem** (Fase 3 do roadmap de pesquisa); suas interfaces serão
definidas junto da primeira implementação real, para não congelar contratos sem uso.

### Memory Management
`MemoryManager`: offload `none/model/sequential` (automático por VRAM: <6.5 GB sequential, <20 GB model; CPU só
`none`; limiares são fonte única, usados também por `select_profile`), estatísticas (alocado/reservado/livre/pico
na GPU; RSS do processo e RAM do sistema na CPU), `track()` para pico por execução, `cleanup()` (gc +
`empty_cache`). VAE tiling continua configuração do modelo. Quantização/otimização de atenção: não implementadas.

### Precision Management
`PrecisionManager.resolve(requested, device, allowed=spec.precisions)`. `auto`: bf16 na GPU que suporta, senão
fp16; fp32 na CPU. CPU aceita fp16/bf16 (executado ponta a ponta no modelo minúsculo, torch 2.14). Aliases antigos
(`bfloat16`, `float16`) aceitos. INT8 etc. só quando implementados e medidos.

### Deployment Architecture
Research = PyTorch eager (atual). Production (Fase 18) começará como PyTorch com config reproduzível; ONNX/TensorRT
entram como novos `Runtime` + `ModelSpec.runtimes` apenas depois de exportação validada numericamente. Status:
ONNX **UNSUPPORTED**, TensorRT **UNSUPPORTED** (sem export validado; sem GPU NVIDIA). Multi-GPU: seleção por índice
implementada; execução distribuída (DDP/FSDP) não, e não será criada sem hardware e necessidade.

## Pipeline de dados implementado (2026-09-24)

```text
                         ┌───────────────────────── preprocessing/pipeline.py ─────────────────────────┐
motion.mp4 ─► decode ─►  │ BodyExtractor (MediaPipe Pose)  → body_motion.pt  + body_preview.mp4       │
                         │ FaceExtractor (Face Landmarker) → face_motion.pt  + face_preview.mp4       │
                         │ HandExtractor (Hand Landmarker) → hand_motion.pt  + hands_preview.mp4      │
                         │   (lado da mão = punho do corpo mais próximo)                               │
                         │ render OpenPose-18 + mãos + pontos de rosto → pose_openpose.mp4            │
                         └─────────────────────────────────────────────────────────────────────────────┘

reference.png ─► letterbox(W×H) ─┐
pose_openpose.mp4 ─► 4k+1 frames ├─► WanVACEPipeline (Wan2.1-VACE-1.3B, Diffusers)
prompt ─► UMT5 (CPU, 1×) ─► cache┘      offload + VAE tiling  ─► output.mp4 / side_by_side.mp4
```

## Avaliação implementada (ADR-007)

`benchmark/v1.draft.yaml` → preparação das mídias → lock com hashes → geração
sequencial pelo baseline → re-extração de driver/output → métricas por caso →
comparação de relatórios → revisão humana. Comando: `scripts/benchmark.py`.

Componentes `evaluation/{protocol,motion,benchmark}.py`: Python 3.12 testado em
Windows; NumPy, PyTorch (leitura segura dos tracks), MediaPipe, OpenCV,
imageio-ffmpeg e PyYAML já existentes. Avaliação em CPU sem CUDA. Geração exige
CUDA e pesos do baseline; ainda não testada com pesos reais. Linux não verificado.

## Módulos treináveis — implementados, testados em CPU, NÃO treinados (2026-09-24)

| Módulo | Arquivo | Contrato verificado |
|---|---|---|
| Features de movimento | `models/motion/features.py` | corpo (33x6), rosto (52 blendshapes + rot6d), mãos (2x21x4) a partir dos `.pt` |
| Encoders corpo/rosto/mãos + pooling 4k+1 -> k+1 + atenção temporal | `models/motion/encoders.py` | tokens por frame latente, máscaras de streams ausentes |
| Identity Encoder v0 (N vistas -> K tokens globais) | `models/identity/encoder.py` | invariante à ordem das vistas; vistas mascaradas não influenciam |
| Fusão (tipo por fonte, frame_index, máscara) | `models/fusion/condition.py` | concatena motion + identity |
| Motion Adapter (cross-attn por bloco, frame-local; caminho denso opcional), zero-init, via hooks | `models/adapters/motion_adapter.py` | **saída bit-idêntica ao backbone no passo 0**, inclusive no pipeline VACE completo (SHA do baseline) |
| Pilha completa | `models/adapters/stack.py` | `MovaConditioning.for_transformer(...)` |

Achado do teste ponta a ponta: o VACE prefixa a referência como um frame latente extra (`reference_frames=1`);
esses frames só enxergam tokens de identidade. É assim que blendshapes e mãos 3D chegam ao modelo: hoje o baseline
só recebe o esqueleto desenhado. Nenhum modelo MOVA foi registrado no registry; isso só acontece com um checkpoint
treinado e avaliado.

## Modelo treinável planejado (desenho original)

```text
Reference image(s) ─► Identity Encoder ─► identity tokens ───────────────────────┐
body_motion.pt  ─► Body Motion Encoder  (conv 3D sobre pose renderizada ou MLP)  │
face_motion.pt  ─► Face Motion Encoder  (blendshapes + head rot6d → tokens)      │
hand_motion.pt  ─► Hand Motion Encoder  (21×2×3 → tokens)                        │
                     └──► Projection ─► Temporal Attention ─► Motion Fusion       │
                                                  │                               │
                                   Motion Adapter (residual zero-init por bloco)  │
                                                  ▼                               ▼
                               Wan2.1 DiT 1.3B CONGELADO  ◄── Identity Injection
                                                  ▼
                                            Wan-VAE decode
```

Treináveis inicialmente: Motion Encoder, Projection, Motion Adapter. Backbone congelado.

### Contratos que os módulos futuros devem respeitar
- Latentes Wan: `(B, 16, (T-1)/4+1, H/8, W/8)`; tokens após patch 1×2×2; dim oculta 1536 (1.3B), 30 blocos.
- `num_frames = 4k+1`; H, W múltiplos de 16.
- Adapter com saída inicializada em zero → no passo 0 o modelo é idêntico ao backbone.
- Entradas de movimento vêm dos `.pt` descritos em `docs/pipeline.md` (FORMAT_VERSION=1).
