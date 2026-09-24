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

---

## ADR-007 — Benchmark versionado e avaliação sem promoção automática

### Decision
Manifesto YAML de casos de teste + lock JSON com SHA-256 das mídias e condições.
Preparar vídeos com duração/FPS/resolução fixos antes de gerar ou avaliar.
Executar casos sequencialmente pelo baseline existente, depois re-extrair
movimento com MediaPipe e comparar tracks crus. Não alterar o backbone.

### Why
Sem entradas fixas e cobertura explícita, uma mudança de dados ou detecções
ausentes pode parecer melhoria. Separar comparabilidade, execução e qualidade
evita promover uma versão só porque não houve erro de código.

### Constraints
Mídias reais e CUDA ausentes; manifesto inicial tem 20 vagas, não 20 vídeos.
PCK@0.1 com torso/palma é diagnóstico experimental, sem threshold perceptual
validado. Identidade e qualidade continuam dependendo de avaliação futura.

### Consequences
Mudanças no benchmark ou avaliador exigem reavaliar ambas as versões.
Dados insuficientes retornam null; casos falhos não desaparecem da comparação.
compare só produz deltas e REQUIRES_REVIEW, sem trocar checkpoints.
O loader legado não fixa revisão HF; registrar essa limitação até corrigi-la.

### Date
2026-09-24

---

## ADR-008 — Reprodutibilidade: revisão fixa, cache verificado, pré-checagem e retomada

### Decision
1. `configs/baseline.yaml` fixa `model.revision` num commit exato do Hub (`ec4d2cb0…`); todos os
   `from_pretrained` usam essa revisão com `local_files_only=True`. O download, quando autorizado, é explícito
   e restrito aos arquivos do manifesto.
2. O manifesto (`configs/model_manifests/<repo>@<sha>.json`, versionado) lista os 17 arquivos necessários com
   tamanho e hash (SHA-256 LFS ou SHA-1 de blob git). O cache só é "completo" se todos existirem com o tamanho
   certo; `--verify-hashes` confere o conteúdo. Substitui a checagem apenas por `model_index.json`.
3. Todo `run.json` grava versões dos pacotes, Python, commit e se a árvore git estava suja, além das diferenças
   contra `requirements.lock.txt` (versões exatas verificadas; tag `+cpu/+cuXXX` ignorada).
4. Pré-checagem de disco/RAM/VRAM (`common/resources.py`) recusa antes de baixar ou carregar qualquer coisa;
   `--skip-resource-check` existe, mas fica registrado no run.
5. `benchmark generate/evaluate` salvam estado atômico após cada caso e aceitam `--resume`. A retomada é recusada
   se lock, versões de pacotes, código de geração ou avaliador mudaram, ou se um vídeo concluído foi alterado.
6. Métricas `motion-v2`: trajetória global da raiz, escala aparente, rotação da cabeça (geodésica) e orientação
   do tronco (yaw 3D), absolutas e relativas ao primeiro frame.
7. Página de revisão (`review/index.html`) com vídeo lado a lado e alertas de revisão por caso; alertas não são
   veredito e não promovem modelos (mantém ADR-007).

### Why
Sem revisão fixa, o mesmo config pode carregar pesos diferentes; `model_index.json` presente não garante os
19 GB restantes; iniciar sem recursos desperdiça horas e pode corromper o cache; benchmarks longos param; a
normalização por raiz/torso esconde "andar no lugar" e giros; revisão manual precisa de contexto visual.

### Constraints
Requisitos de RAM/VRAM são estimativas a partir do tamanho dos arquivos — recalibrar com EXP-001.
Métricas de trajetória exigem quadris e ombros visíveis; caso contrário retornam null.

### Consequences
Mudar a revisão do modelo exige novo manifesto e novo experimento. `METRIC_VERSION` passou a `motion-v2`:
relatórios motion-v1 não são comparáveis com motion-v2 (a assinatura do avaliador muda).

### Date
2026-09-24

---

## ADR-009 — Camadas Core / Model / Runtime e CLI única

### Decision
O sistema é dividido em camadas com dependência em um só sentido:
`common ← runtime ← models ← core ← {mova CLI, scripts, API futura}`.
1. `runtime/`: interface `Runtime` + `PyTorchRuntime` (único backend implementado), `DeviceManager`
   (único lugar que consulta a GPU), `PrecisionManager` (fp32/fp16/bf16), `MemoryManager` (offload, estatísticas,
   limpeza), `RuntimeManager` (lookup por nome). ONNX/TensorRT são listados como **não implementados** e pedir
   por eles gera `RuntimeNotAvailableError` — não existem classes vazias.
2. `models/`: interface `MotionModel` (configure → weights_status/fetch_weights → resource_checks → load →
   generate → unload), `ModelSpec` (metadados únicos: runtimes, devices, precisões, capabilities, requisitos,
   pesos, status de verificação) e registry com aliases. `inference/baseline_vace.py` mantém só a numérica.
3. `core/`: precedência de config (`configs/runtime.yaml` < `runtime:` do run config < flags), validação de
   capabilities antes de qualquer trabalho pesado, serviço de inferência único, info e preprocess.
4. `mova` (console script): `info`, `infer`, `preprocess`, `benchmark`, `evaluate`, `test`, `train` (honesto:
   não implementado). `scripts/inference_baseline.py` e `extract_motion.py` viram wrappers da CLI.
5. Erros estruturados em `common/errors.py` com `code` estável (para CLI e API).
6. Layout plano mantido (sem mover pacotes para `mova/…`): mover quebraria imports, scripts e o hash de código
   do benchmark sem ganho funcional. `common/env.py` continua como fachada compatível sobre o `DeviceManager`.

### Alternatives Considered
Reescrever em `mova/{core,runtime,...}`; criar classes ONNX/TensorRT/ROCm como placeholders; criar interfaces
para encoders de identidade/movimento que ainda não existem. Rejeitados: custo de migração sem uso real, ou
funcionalidade fictícia.

### Why
Tirar `pipe.to("cuda")` e `torch.cuda.*` do código de modelo; permitir escolher GPU N, CPU ou ROCm por config;
mesma lógica para CLI/scripts/API; incompatibilidades explicadas antes de carregar 19 GB.

### Evidence
Saída fp32 **bit-idêntica** ao baseline da Fase 0 (`benchmark/baseline` vs `benchmark/regression`, SHA-256
`a3cbab52…`); 129 testes passando em CPU. Caminho CUDA/offload não executado (sem GPU) — UNVERIFIED.

### Consequences
Um backbone novo = um módulo em `models/backbones/` que chama `register_model`; core/CLI não mudam (testado com
um modelo temporário). `model.dtype`/`model.offload` nos run configs estão obsoletos (migrados com aviso).
Retomar um benchmark iniciado antes desta mudança é recusado (o hash do código de geração mudou) — esperado.

### Date
2026-09-24

---

## ADR-010 — Linguagens e backends

### Decision
- **Python** é a linguagem da IA (modelos, runtime, core, CLI, API).
- **CUDA** via PyTorch; kernels próprios (C++/CUDA em `native/`) só com gargalo comprovado por profiling, com
  API Python, fallback, testes, benchmark e documentação. Hoje: nenhum; `native/` não foi criado.
- **Rust** não faz parte do core de IA (possível apenas num app desktop futuro, ex.: Tauri).
- **TypeScript/React** apenas para interface (sem lógica de IA). Hoje não existe frontend.
- **ONNX Runtime / TensorRT**: backends opcionais de deployment, só depois de exportação validada numericamente
  contra o PyTorch e benchmark real. Hoje: UNSUPPORTED.
- Suporte a hardware é declarado por ambiente testado (NVIDIA, CPU, AMD), nunca "suporta tudo".

### Why
Ecossistema (PyTorch, Diffusers), velocidade de pesquisa e manutenção; o gargalo real ainda não foi medido
(EXP-001 bloqueado), então não há base para código nativo.

### Date
2026-09-24
