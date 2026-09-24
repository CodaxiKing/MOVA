# Architecture Decision Records

Nunca apagar um ADR. Para reverter, criar um novo ADR que referencia o antigo.

---

## ADR-001 — Backbone Selection

### Decision
Backbone de desenvolvimento: **Wan2.1 DiT 1.3B** (congelado). Adapters próprios serão treinados sobre ele.

### Alternatives Considered
Wan2.2-Animate-14B, Wan-Animate-2 (14B), Wan2.1-Fun-1.3B-Control, UniAnimate-DiT (14B), MimicMotion (SVD), Moore-AnimateAnyone (SD1.5), MagicAnimate, StableAnimator, Wan2.2-TI2V-5B.

### Why
Maior DiT de vídeo Apache 2.0 que cabe e treina (LoRA/adapters) em 8 GB; tokens escalam bem para 256px/17 frames (~1 280 tokens); ecossistema de treino (DiffSynth, VideoX-Fun); mesma família dos modelos SOTA open-source (Wan-Animate), permitindo upgrade. Detalhes: `docs/research/backbone_selection.md`.

### Constraints
RTX 3060 8 GB; 16 GB RAM; sem treino do backbone.

### Date
2026-09-24

### Consequences
Qualidade limitada pelo 1.3B (esperado). Código de adapter deve seguir convenções Wan (VAE 4×/8×, patch 1×2×2, `num_frames = 4k+1`).

---

## ADR-002 — Baseline Model

### Decision
Baseline da Fase 1: **Wan2.1-VACE-1.3B via Diffusers `WanVACEPipeline`** (pose video OpenPose-style + reference image). Baseline B opcional: Wan2.1-Fun-V1.1-1.3B-Control.

### Alternatives Considered
Wan2.2-Animate-14B (grande demais), MimicMotion (licença SVD, U-Net, sem treino), Fun-Control (empate técnico, mas fora do Diffusers).

### Why
Faz reference+motion→vídeo sem treino, no mesmo backbone que vamos adaptar; mecanismo aditivo por camada = mesmo contrato do nosso Motion Adapter → comparação justa; suporte nativo a offload/VAE tiling no Diffusers.

### Constraints
Download de 19.04 GB (exige autorização do usuário); ≥ 25 GB livres.

### Date
2026-09-24

### Consequences
Pose de entrada precisa estar no estilo OpenPose; nosso render a partir de MediaPipe é uma aproximação (ver ADR-003, EXP-004).

---

## ADR-003 — Motion Extraction Stack

### Decision
**MediaPipe Tasks** (Pose 33 + world 3D, Face 478 + 52 blendshapes + matriz de cabeça, Hand 21×2 + world 3D) para extração estruturada; corpo/rosto/mãos salvos **separadamente** (`body_motion.pt`, `face_motion.pt`, `hand_motion.pt`). Render OpenPose-18 derivado do MediaPipe para o baseline. DWPose (ONNX) fica como alternativa a avaliar.

### Alternatives Considered
DWPose (133 pts, formato nativo dos modelos Wan, mas sem 3D, sem blendshapes e mais lento em CPU); SMPL/SMPL-X (licença não comercial); OpenPose original (licença não comercial).

### Why
Apache 2.0, roda rápido em CPU (~20 ms/frame para corpo, medido), dá 3D e expressão facial prontos — permite a Fase 2 inteira na máquina sem GPU.

### Constraints
Máquina de desenvolvimento atual sem GPU.

### Date
2026-09-24

### Consequences
Mapeamento MediaPipe→OpenPose-18 é aproximado (sem pés/olhos idênticos ao DWPose). Mãos: atribuição esquerda/direita pelo punho do corpo mais próximo (MediaPipe assume imagem espelhada).

---

## ADR-004 — VRAM/RAM Strategy for Inference

### Decision
Encoder de texto UMT5-XXL roda **uma vez em CPU**; embeddings do prompt são **cacheados** em `checkpoints/embeds/`; o pipeline de geração é carregado **sem** o encoder de texto. Offload automático por perfil (`common/env.py`): <6.5 GB → sequential; <10 GB → model offload; VAE tiling ligado. Perfil 8 GB: 256px, 17 frames.

### Alternatives Considered
Carregar tudo com `enable_model_cpu_offload` (UMT5 bf16 ~11 GB compete com a RAM de 16 GB); GGUF/quantização do encoder (mais dependências).

### Why
O DiT 1.3B não é o gargalo; o UMT5 é. Isolar seu uso reduz pico de RAM e VRAM. No treino, o encoder nem é carregado.

### Date
2026-09-24

### Consequences
Mudança de prompt exige nova codificação (uma vez). Thresholds dos perfis são iniciais e devem ser recalibrados com medições reais (EXP-001).

---

## ADR-005 — Environment

### Decision
Python **3.12** (3.11 não está instalado na máquina; todas as dependências suportam 3.12). venv em `.venv/`. PyTorch instalado conforme hardware (CPU aqui; cu12x na máquina da RTX 3060).

### Date
2026-09-24

### Consequences
Wan-Animate-2 recomenda 3.11 — irrelevante enquanto não for executado localmente.

---

## ADR-006 — Project License (PROVISIONAL)

### Decision
Código do MOVA sob **Apache-2.0** — **provisório, aguardando confirmação do usuário**.

### Why
Compatível com todas as dependências e modelos escolhidos (todos Apache 2.0).

### Date
2026-09-24
