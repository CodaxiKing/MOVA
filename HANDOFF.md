# HANDOFF

## Ajuste de prévia do Estúdio — 2026-09-24

Em `design/canvas/project/Main.dc.html` e `web/canvas-live.js`, as duas prévias do movimento perderam `autoplay`/`loop`. `scripts/build_web.py` regenerou `web/estudio.html`. Com "Usar exemplos" no navegador, ambas as tags `<video>` mostraram `paused=true` e `autoplay=false`. O arquivo continua selecionado para a geração. `pytest -q --tb=line`: **215 passed**. Mudança apenas visual, sem alteração do core.

## Correção do site pelo layout do canvas — 2026-09-24

O site genérico da sessão anterior foi substituído pelo layout original de `design/canvas/project`. `scripts/build_web.py` gera as seis telas e carrega `canvas-live.js` antes de `dc-runtime.js`; o runtime chama os hooks de ligação. `web/server.py` fornece runs, mídia de entrada/saída, exemplos locais, jobs e revisão visual por run (`web_review.json`). Páginas inspecionadas no navegador local: Início, Estúdio, Movimento, Resultados, Experimentos e Ambiente; console sem erros. `web/app.js`/`app.css` antigos foram removidos. `pytest -q --tb=line` fora do sandbox: **215 passed**. Reiniciar o servidor após mudanças em `web/server.py`. A inferência Wan via Estúdio ainda depende de validação com mídia/recursos; não foi repetida só para testar layout.

## Site funcional local — 2026-09-24

Foi criado `web/server.py` (stdlib, sem dependência FastAPI) e `web/app.js`/`app.css`; `scripts/build_web.py` injeta a interface ativa nas seis páginas. Rodar `.venv/Scripts/python web/server.py` e abrir `http://127.0.0.1:8000`. Ambiente, Experimentos, Movimento, Estúdio e Resultados consultam/chamam o core real. O backend recusa download de pesos e serve apenas em localhost. API info e runs verificadas por HTTP; fluxo de geração com mídia real ainda pendente. Há alterações preexistentes em `inference/baseline_vace.py` e `tests/test_wan_weights.py`, não relacionadas a este trabalho.

O primeiro `pytest -q` desta sessão teve 121 passed, 94 errors porque o diretório temporário recebeu `PermissionError`; a repetição com `--basetemp=.pytest_tmp` teve o mesmo problema. A suíte fora do sandbox passou: **215 passed**. Upload de `control.mp4` pela API + extração passou e criou `20260924-213136-extract`. Próximo: validar inferência via navegador e EXP-001 conforme RAM/VRAM disponível.

## Sessão atual — primeira máquina com GPU, 2026-09-24, Claude Code (branch `main`)

Pedido: "inicie o projeto todo", depois "corrija e baixe o Wan". Máquina NOVA, fora dos docs até agora: desktop
i5-10400F + RTX 2060 SUPER 8 GB (cc 7.5), 16 GB RAM com ~2–3 GB livres. Nada commitado.

Feito:
- `.venv` com Python 3.12.10 (winget, escopo de usuário) + torch 2.14.0+cu130 + lock exato; modelos MediaPipe.
- Caminho GPU corrigido (embeddings de prompt no device de execução; `place` com offload só em pipelines) e
  verificado com o tiny em `tests/test_cuda.py`. 214 testes passando, 0 skipped.
- Bit-exato: referência por capacidade de CPU (`benchmark/baseline/reference.py`); prova: script congelado no
  commit 8284436 reproduz `d220ac…` nesta CPU AVX2.
- Pesos do Wan baixados via `WanVACEModel.fetch_weights(verify_hashes=True)` no cache HF do usuário.
- Site do canvas em `web/` (`python scripts/build_web.py`; servir com `python -m http.server -d web`).

Não feito / bloqueado: EXP-001. Faltam `assets/reference/maya.png` e `assets/motion/dance.mp4`, e ~12 GB de RAM
livre para o UMT5 (uma vez; depois o prompt fica em `checkpoints/embeds`). Risco: bf16 em Turing.

Próximo: mídias → fechar programas → `mova infer --model wan --device cuda:0 --reference … --motion … --output …`
(bf16; se lento/instável, repetir com `--precision fp16`) → registrar VRAM/tempo no EXP-001 e na matriz.

## Sessão atual — qualidade do motion control em CPU, 2026-09-24, Claude Code

Pedido: implementar tudo o que é testável em CPU e melhora o motion control (métricas, retargeting, sinal, Fase 3,
dados). Branch **`feat/motion-quality`** (criada a partir de `refactor/runtime-architecture`, sem upstream, sem
push, sem PR). 204 testes passando. Nada treinado, nada baixado, nenhum vídeo real gerado.

Feito (commits e4f4d91 → 9f79376 + docs): métricas identity-v1/temporal-v1 + GSB cego; retargeting; One-Euro,
pós-processamento de mãos, controle a partir das tracks, tracks v2; encoders, identidade, fusão e Motion Adapter
zero-init; scaffolding de treino e `mova train --smoke`; auditoria de licenças, registro de fontes, validação de
pares, build do manifesto e `benchmark intake`. EXP-003 e EXP-006 (sintéticos). ADR-011..014.

Achados que valem para a GPU:
- O `WanVACEPipeline` prefixa a referência como um frame latente extra → `reference_frames=1` no adapter.
- O `forward` do `WanVACETransformer3DModel` exige controle → treino com controle nulo (escala 0).
- Suavizar o controle custa atraso em dança rápida (EXP-006); padrão continua sem suavização.
- A entrada atual do encoder de corpo (2D normalizado) é a mais sensível a ruído no EXP-003 sintético.

Não tocado: pasta `design/` (canvas criado fora desta sessão, não rastreado).

Próximo: na RTX 3060, EXP-001 → gerar o benchmark → `benchmark evaluate` com identity/temporal → primeira sessão GSB
(baseline vs baseline + retarget) → só então `mova train` real.

## Sessão atual — evolução arquitetural (Runtime/Model/Core/CLI), 2026-09-24, Claude Code

Pedido: "prompt mestre" de evolução arquitetural (runtime desacoplado, device/precision/memory managers, interface
de modelo, registry, capabilities, CLI/API sobre o mesmo core), sem quebrar o que funciona e sem funcionalidade
fictícia. Branch **`refactor/runtime-architecture`** (sem upstream; não enviado ao GitHub, sem PR).

Commits: 36b5a08 (docs de pesquisa pendentes da sessão anterior), 8284436 (auditoria + baseline), 9b22b9f
(runtime), 059cfb5 (model/registry/core/CLI), acec055 (fix CLI), + docs.

Feito e verificado nesta máquina (CPU): ver matriz em STATUS.md. Resumo: 129 testes; saída fp32 bit-idêntica ao
baseline pré-refatoração; `mova infer --model tiny` ponta a ponta com MediaPipe real; todos os erros de
compatibilidade saem antes de carregar qualquer coisa.

NÃO verificado: nada com GPU. `TorchDeviceProbe.accelerator`, `PyTorchRuntime.place` com offload, stats de VRAM e
`WanVACEModel.build/encode_prompts` com pesos reais nunca rodaram. API, plugins de encoders, produção, ONNX,
TensorRT, multi-GPU e nativo: não iniciados/bloqueados (motivos na matriz).

Como continuar:
1. `git switch refactor/runtime-architecture`; `.venv/Scripts/python -m pip install -e . --no-deps` (comando `mova`).
2. Na RTX 3060: `mova info` → conferir GPU/VRAM/cc/bf16; `mova test -q`; `mova infer --model tiny --device cuda`
   (primeira validação real do runtime na GPU; barato); depois EXP-001 com `mova infer --model wan` e
   `--allow-download` **somente com autorização** (19.04 GB).
3. Atualizar `ModelSpec.verification` e a matriz do STATUS com o que for medido.
4. API (Fase 11): precisa decidir a dependência FastAPI; deve chamar `core.inference.run_inference`.

Cuidados: não reintroduzir `torch.cuda`/`"cuda"` fora de `runtime/`; não criar classes ONNX/TensorRT/ROCm vazias;
`benchmark generate --resume` de runs antigos é recusado (hash do código mudou) — iniciar run novo.

## Sessão atual — pesquisa Kling 3.0 + auditoria do repositório, 2026-09-24

Objetivo do usuário: "verificar como o Kling Motion Control 3.0 funciona e o que
temos/falta". Sem alterações de código; apenas docs de pesquisa e auditoria.

- Auditoria do repo (subagent): 16 commits, main == origin/main; `models/` e
  `training/` vazios (só `__init__.py`); `datasets/` e `assets/` sem conteúdo;
  0 pesos Wan em cache; 0 gerações reais; 49 funções de teste.
- Pesquisa Kling: paper 2603.03160 **não tem seção Method** (FACT, lido
  integralmente); detalhes de produto no guia oficial
  kling.ai/quickstart/motion-control-user-guide (Element Binding = só facial,
  entradas 3–30 s, saída 720p/1080p, preços 9/12 credits/s, sujeito único).
  Comparação independente: Wan-Animate-2 §5 = "comparable performance".
- `docs/research/kling_analysis.md` ganhou §6 (produto 3.0) e fontes novas;
  `sources.md` atualizado; CHANGELOG e TODO (itens PE e GSB) atualizados.

Validação: `.venv/Scripts/python -m pytest -q -p no:cacheprovider --basetemp outputs/test-klingresearch-20260924`
→ **64 passed in 47.63s**, CPU. Nenhum download; nenhum peso tocado.

Gap consolidado (detalhe na resposta da sessão): baseline real bloqueado
(sem GPU/autorização/mídia) e Fase 3 (`models/`, `training/`) 100% planejada.
Próxima ação inalterada: RTX 3060 + autorização 19 GB + mídia → EXP-001.

Histórico abaixo; STATUS.md contém o estado consolidado.

## Continuação — Claude Code, 2026-09-24 (reprodutibilidade e revisão)

Pedido do usuário: fixar pesos/dependências, verificar cache completo, checar RAM/disco, retomar benchmarks,
medir trajetória global e rotação da cabeça, e facilitar a revisão visual. Tudo implementado (ADR-008).

Arquivos novos: `common/provenance.py`, `common/resources.py`, `evaluation/review.py`, `requirements.lock.txt`,
`configs/model_manifests/Wan-AI--Wan2.1-VACE-1.3B-diffusers@ec4d2cb0….json`,
`tests/test_reproducibility.py`, `tests/test_benchmark_resume_review.py`.
Alterados: `common/{hf_utils,experiment}.py`, `configs/baseline.yaml`, `inference/baseline_vace.py`,
`scripts/{inference_baseline,benchmark,check_model_size}.py`, `evaluation/{motion,benchmark}.py`, docs.

Verificado: `pytest -q` → **64 passed** (CPU). `check_model_size.py` → revisão ec4d2cb0, 0/17 no cache.
Gate do baseline para em "Missing 19.04 GB" sem baixar. Pré-checagem com números reais recusaria (disco 17.8 <
22 GB; RAM 1.7 < 13 GB). E2E real com MediaPipe + página de revisão aberta no navegador.

Não verificado: nada disso rodou com GPU/pesos reais. Estimativas de RAM/VRAM precisam de EXP-001.
Não rode `--allow-download` nesta máquina (disco insuficiente e sem autorização do usuário).


## Sessão atual — benchmark e avaliação, 2026-09-24

Objetivo autorizado: criar o ciclo de baseline real + benchmark + métricas.
Implementada a infraestrutura possível em CPU; geração real bloqueada por
hardware, pesos e mídia. Não houve treinamento ou download.

Arquivos principais novos: evaluation/{motion,protocol,benchmark}.py,
scripts/benchmark.py, tests/test_benchmark.py, benchmark/ (manifesto de 20 vagas,
guia e templates), docs/experiments/EXP-005.md. Docs sincronizados, ADR-007.
Arquitetura: mesma extração e backbone; camada de avaliação independente.

Validação final: `.venv/Scripts/python -m pytest -q -p no:cacheprovider --basetemp outputs/test-benchmark-release-20260924`
→ **51 passed in 29.42s**, CPU. Testes do generate usam baseline simulado;
extração e avaliação MediaPipe foram executadas realmente em fixtures sintéticas.
Avaliação facial foi repetida. Astronaut não tem quadris visíveis suficientes:
PCK corporal null é correto. Não baixar o limiar para forçar pontuação.

Preflight: `python scripts/benchmark.py check` → BLOCKED, 20 vagas, 140 pendências.
Mídias ainda não fornecidas. CUDA False, RAM disponível 0.94 GB, disco 16.66 GB.
Pergunta enviada ao usuário solicitando caminhos de mídias autorizadas e aviso
quando estiver na máquina RTX 3060; ainda sem resposta nesta sessão.

Próximos passos:
1. Ler benchmark/README.md e docs/evaluation.md para contratos e limitações.
2. Obter mídias autorizadas; preparar clipes e preencher fontes/licenças/identity_id.
3. Na GPU, executar EXP-001 primeiro; pesos grandes ainda exigem autorização.
4. Congelar benchmark, generate, evaluate, repetir e compare; revisar visualmente.
5. Usar falhas medidas para planejar adapter; manter test separado de treino/validação.

Não promover checkpoints por PCK isolado. compare exige mesmos hashes/condições
e retorna REQUIRES_REVIEW. null não é erro zero. Arquivos ausentes não podem
ser removidos silenciosamente. Loader legado não fixa revisão HF; corrigir
antes de afirmar reprodutibilidade entre caches diferentes. Linux não testado.

Histórico anterior abaixo; STATUS.md contém o estado consolidado atual.


## Continuação — Codex, 2026-09-24

Objetivo: auditar e continuar a preparação do baseline sem avançar para adapters
antes da geração real, conforme prompt mestre. Estado inicial: main/3174705, limpo.

Concluído: `evaluation/video.py`, integração em `scripts/inference_baseline.py`,
`tests/test_video_validation.py`, smoke VACE com encoding e validação.
Documentação sincronizada e PROJECT_MEMORY.md criado (estava ausente).
Nenhum download, treino ou alteração do backbone.

Teste final: `.venv/Scripts/python -m pytest -q -p no:cacheprovider --basetemp outputs/test-final-20260924`
→ **35 passed in 12.55s**, CPU, fora do sandbox devido a PermissionError nas pastas
temporárias. Use uma pasta temporária nova em cada execução. check_env confirmou
CUDA False, torch 2.14.0+cpu, RAM livre 0.99 GB.

Próximos passos: obter acesso à máquina RTX 3060, autorização para pesos e mídia;
executar EXP-002/EXP-001 conforme roteiro abaixo. Depois comparar controle e só
então prototipar adapter. Não declarar baseline PASS com testes de pesos aleatórios.
A validação de vídeo mede integridade e não certifica fidelidade visual.

## Handoff anterior (histórico)


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
