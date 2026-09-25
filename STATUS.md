# Project Status

## Site local ligado ao core (2026-09-24)

`web/server.py` serve as seis abas em `127.0.0.1:8000`. Ambiente consulta `core.info`, Experimentos lê `run.json`, Movimento chama `core.preprocess.run_extraction`, Estúdio chama `core.inference.run_inference`, e Resultados exibe MP4 registrado. Wan não baixa pesos pela interface. Rotas da home, `/api/info`, `/api/runs` e mídia responderam 200 em teste local; job inválido foi registrado como failed. Upload + extração via API passaram e criaram `20260924-213136-extract`. `pytest -q --tb=line`: **215 passed** fora do sandbox; a execução restrita teve PermissionError no diretório temporário do pytest. Inferência real pelo navegador ainda não foi executada. Ver `web/README.md`.

## Primeira máquina com GPU — setup, correção do caminho CUDA, pesos do Wan (2026-09-24, Claude Code, `main`)

Máquina: desktop i5-10400F (AVX2) + **RTX 2060 SUPER 8 GB** (cc 7.5), 16 GB RAM (~2–3 GB livres), Python 3.12.10,
torch 2.14.0+cu130, demais pacotes = `requirements.lock.txt`. Testes: **214 passed** (204 + 10 CUDA novos), 0 skipped.
Nenhum vídeo real gerado.

| Item | Status | Evidência |
|---|---|---|
| Caminho GPU (runtime place/offload, precisões, VRAM stats) com tiny | **PASS** | `tests/test_cuda.py`: fp32/bf16/fp16 × offload none/model/sequential; fp32 GPU = CPU (< 1e-3) |
| Bug: embeddings de prompt ficavam na CPU com offload → `mat1 is on cpu` no transformer | **FIXED** | `inference/baseline_vace.py` move para `pipe._execution_device`; no-op em CPU (bit-exato mantido) |
| Bug: `PyTorchRuntime.place` chamava `enable_model_cpu_offload` em não-pipelines | **FIXED** | `hasattr` antes do hook, senão `.to()` (como o docstring já dizia) |
| Testes que assumiam `device=auto` → cpu | **FIXED** | testes de CPU fixam `runtime.device=cpu`; GPU em `tests/test_cuda.py` |
| Regressão bit-exata no i5-10400F | **PASS** com referência AVX2 | o script congelado no commit 8284436 gera `d220ac…` aqui: diferença é de kernel SIMD, não de código (`benchmark/baseline/reference.py`) |
| Pesos Wan2.1-VACE-1.3B @ ec4d2cb | **PASS** — COMPLETE, 17/17 arquivos, SHA-256 conferidos | `fetch_weights(verify_hashes=True)` → `COMPLETE (17/17 files; hashes checked: True)` |
| EXP-001 (baseline real) | **BLOCKED** | sem `maya.png`/`dance.mp4`; UMT5 precisa de ~12 GB RAM livre |
| bf16 em Turing | **UNVERIFIED** (risco) | cc 7.5 sem bf16 nativo; perfil escolhe bf16 |

## Qualidade do motion control em CPU (2026-09-24, Claude Code, branch `feat/motion-quality`)

Tudo abaixo roda e foi testado em CPU, sem os 19 GB e sem GPU. **Nada foi treinado e nenhum vídeo real foi gerado**:
isto melhora o que medimos e o que o modelo poderá usar, não a qualidade já demonstrada.
Testes: **204 passed** (antes: 129). Máquina: i7-1165G7, sem CUDA, torch 2.14.0+cpu.

| Item | Status | Evidência |
|---|---|---|
| Identidade identity-v1 (geometria facial rígida + cor Lab rosto/torso + embedding opcional) | **PASS (sintético + foto real)** / calibração em vídeo gerado **UNVERIFIED** | astronaut: mesma pessoa 0.009, rosto +29 % 0.055, recolorido ΔE 35 vs 1.5; invariância exata a pose/escala testada |
| Temporal temporal-v1 (warp error com oclusão, flicker, fluxo, razão vs driver) | **PASS (sintético)** | suave 0.0009, ruído 0.05, flicker 0.157, congelado 0 + fluxo ≈ 0 sinalizado |
| GSB pareado cego (5 eixos Kling + mãos + overall, (G+S)/(B+S)) | **PASS (ferramenta)**; nenhuma sessão real | cegamento, pontuação do ponto de vista do candidato, validação de entradas |
| Integração benchmark evaluate / compare / review | **PASS** | avaliação real com MediaPipe inclui identity/temporal; review mostra tabelas e alertas |
| Retargeting por comprimento de ossos (+ mãos, rosto, âncora na referência) | **PASS (geometria sintética)** / com personagem real **UNVERIFIED** | comprimentos 3D = referência, direções = driver, 2D = projeção do 3D, identidade = no-op |
| One-Euro + pós-processamento de mãos + controle a partir das tracks | **PASS** | controle pixel-idêntico ao antigo nos padrões; trocas E/D e mãos distantes corrigidas em testes |
| EXP-003 (representações) | **PARTIAL** (sintético) | rot6d dos ossos longos: invariante a câmera/corpo, ruído 0.63 vs 5.69 (todos os ossos) |
| EXP-006 (suavização) | **PARTIAL** (sintético) | padrão continua sem suavização: atraso > jitter em movimento rápido |
| Motion/Face/Hand encoders, Identity Encoder v0, fusão, Motion Adapter zero-init | **PASS (forma/invariantes)** / treinado **NÃO** | backbone bit-idêntico no passo 0, inclusive no pipeline VACE (SHA do baseline) |
| Treino: losses, precompute, dataset, trainer, `mova train --smoke` | **PASS (CPU minúsculo)** / treino real **BLOCKED** | overfit, retomada exata, backbone intacto, run.json |
| Licenças + registro de fontes + validação de pares + build do manifesto | **PASS** | InsightFace/LivePortrait/UniAnimate-DiT bloqueados, AIST research-only (fontes primárias) |
| `benchmark intake` (mídia + metadados, depois freeze) | **PASS (sintético)**; benchmark real ainda **BLOCKED** (sem mídia) | intake → freeze funciona com um caso sintético |

Formato dos tracks: `FORMAT_VERSION = 2` (mãos pós-processadas por padrão; cru preservado). Detalhes:
`docs/pipeline.md`, `docs/evaluation.md`, `docs/training.md`, `docs/architecture.md`, ADR-011..014.

## Evolução arquitetural — Runtime / Model / Core / CLI (2026-09-24, Claude Code, branch `refactor/runtime-architecture`)

Máquina: HP ProBook, i7-1165G7, **sem CUDA**, torch 2.14.0+cpu, RAM livre 0.8–1.9 GB. Nenhum download.
Testes: **129 passed** (antes: 64). Detalhes: `docs/architecture.md`, ADR-009/010, `docs/architecture-audit.md`.

| Fase | Entregável | Teste obrigatório | Status |
|---|---|---|---|
| 0 | Auditoria + baseline | Pipeline atual | **PARTIAL** — auditoria + baseline CPU (VACE minúsculo) feitos; baseline real BLOCKED (sem CUDA/pesos/mídia) |
| 1 | Runtime | Inferência via PyTorchRuntime | **PARTIAL** — CPU PASS (bit-exato vs baseline, e2e CLI); CUDA/offload UNVERIFIED |
| 2 | Device Manager | Detecção de device | **PARTIAL** — CPU PASS; lógica multi-GPU/ROCm/sem-bf16 testada com probe simulado; probe CUDA real nunca executado |
| 3 | Model Interface | Modelo via interface | **PASS** — modelo atual pela interface, saída fp32 bit-idêntica |
| 4 | Runtime Config | Configuração externa | **PASS** — runtime.yaml / `--set runtime.*` / flags mudam device e precisão sem tocar código (testado) |
| 5 | Precision Manager | FP32/FP16/BF16 | **PARTIAL** — CPU: os 3 modos e2e, saída finita, tempo registrado; GPU e VRAM UNVERIFIED |
| 6 | Memory Manager | Métricas/controle | **PARTIAL** — política de offload, stats CPU, cleanup e load/unload testados; VRAM antes/pico/depois UNVERIFIED |
| 7 | Capability System | Validação de compatibilidade | **PASS** — runtime/modelo/device/precisão/offload/pacotes/opt-in/pesos/recursos com erro claro antes de executar |
| 8 | Model Registry | Registro/carregamento | **PASS** — 2º backbone registrado em teste e executado sem mudar core/CLI |
| 9 | Plugins/Adapters | Substituição de módulo | **PARTIAL** — encoders corpo/rosto/mãos, identidade, fusão e adapter com interfaces e testes; streams selecionáveis por config; não treinados |
| 10 | CLI | Execução end-to-end | **PARTIAL** — info/infer/preprocess/benchmark/evaluate/test PASS; `train --smoke` PASS (treino real bloqueado); sem API para comparar |
| 11 | API | Inferência via API | **UNVERIFIED** — não iniciada (FastAPI não instalado; core pronto para ser chamado) |
| 12 | Regression Benchmark | Baseline vs nova arquitetura | **PARTIAL** — CPU minúsculo: bit-exato, overhead explicado; modelo real BLOCKED |
| 13 | ONNX | Benchmark real | **BLOCKED** — sem export validado; `onnx` não instalado. UNSUPPORTED |
| 14 | TensorRT | Benchmark real | **BLOCKED** — sem GPU NVIDIA nem export. UNSUPPORTED |
| 15 | Multi-GPU | Execução real | **BLOCKED** — sem hardware; só seleção `cuda:N` implementada |
| 16 | C++/CUDA | Profiling + benchmark | **BLOCKED** — sem profiling do modelo real; nenhum código nativo criado |
| 17 | Portabilidade | Testes por hardware | CPU (x86, Windows 11) **PASS** com modelo minúsculo · NVIDIA **UNVERIFIED** · AMD/ROCm **UNVERIFIED** · Linux **UNVERIFIED** |
| 18 | Production Runtime | Execução reproduzível | **UNVERIFIED** — não iniciada |
| 19 | Documentação | Reprodução por terceiro | **PARTIAL** — docs atualizados; reprodução por terceiro não feita |
| 20 | Validação Final | End-to-end completo | **BLOCKED** — depende de EXP-001 na GPU |

### Evidência por fase concluída ou parcial

```text
Phase: 0 — Auditoria + baseline            Status: PARTIAL
Implemented: docs/architecture-audit.md; benchmark/baseline/capture_tiny_vace.py
Commands: python scripts/check_env.py --cuda-test; pytest -q (64 passed, 27.79s);
          python benchmark/baseline/capture_tiny_vace.py --repeat 3
Results: 3/3 runs SHA-256 a3cbab52523b48c2…, finite, 5×32×32×3
Known Issues: baseline real (Wan2.1-VACE-1.3B) BLOCKED: sem CUDA, 0/17 arquivos (19.04 GB), sem mídia
Evidence: benchmark/baseline/tiny_vace_cpu.json, outputs/arch-audit/env_before.json (local), commit 8284436

Phase: 1/2/5/6 — Runtime, Device, Precision, Memory      Status: PARTIAL (CPU PASS, GPU UNVERIFIED)
Implemented: runtime/{base,device,precision,memory,manager}.py, runtime/backends/pytorch.py, common/errors.py;
             common/env.py e common/resources.py delegam ao DeviceManager
Tests: tests/test_runtime.py (27): auto/cpu/cuda/cuda:N, device ausente/inválido, ROCm, defaults e aliases de
       precisão, bf16 sem suporte, limiares de offload, offload em CPU recusado, tracking de memória real na CPU,
       runtime desconhecido, onnx/tensorrt "not implemented", hooks de offload ligados à GPU escolhida
Metrics: CPU fp32/bf16/fp16 ~0.09 s de geração (modelo minúsculo); VRAM não mensurável aqui
Evidence: `grep torch.cuda` fora de runtime/ → só scripts/check_env.py (diagnóstico); commit 9b22b9f

Phase: 3/4/7/8/10/12 — Model interface, Config, Capabilities, Registry, CLI, Regression
Implemented: models/{base,registry}.py, models/backbones/wan_vace.py, core/{config,capabilities,inference,info,
             preprocess}.py, configs/runtime.yaml, configs/smoke_tiny_vace.yaml, mova/cli.py (console script `mova`)
Tests: tests/test_models_core.py (38): ciclo de vida bit-exato, erros de registry, validação (6 incompatibilidades),
       precedência de config, migração de model.dtype, e2e por precisão, entradas inválidas sem iniciar nada, pesos
       ausentes sem download, 2º backbone plugado, CLI (info JSON/texto, 6 erros estruturados, e2e JSON, repasse de
       flags, train honesto), pipeline completo com extração MediaPipe real
Commands: mova info --model wan; mova infer --model tiny --reference reference.png --motion motion.mp4 --output output.mp4;
          python benchmark/regression/tiny_vace_regression.py
Results: e2e com foto astronaut + vídeo de movimento: corpo/rosto 100% detectados, output.mp4 5 frames 32×32 16 fps
         h264 0.3125 s, finito, validação PASS (2 caminhos independentes). fp32 bit-idêntico ao baseline.
Metrics: regressão CPU: geração igual (~0.09 s); total por run 0.12 → 0.26–0.31 s = gc.collect() no unload (medido)
Known Issues: caminho real (from_pretrained + offload CUDA) nunca executado; mova train não implementado
Evidence: benchmark/regression/tiny_vace_cpu.json, benchmark/regression/README.md; commits 059cfb5, acec055

Next Step: na RTX 3060 — `mova info`, `pytest -q`, EXP-001 via `mova infer --model wan` (com autorização do download)
e repetir benchmark/regression com o modelo real (baseline antes vs depois não se aplica mais: registrar o novo).
```

## Atualização — reprodutibilidade e revisão (2026-09-24, Claude Code)

- Revisão do Wan2.1-VACE-1.3B fixada (`ec4d2cb0…`), manifesto versionado com 17 arquivos e hashes.
- Cache verificado arquivo por arquivo (tamanho; `--verify-hashes` para conteúdo). Nesta máquina: 0/17, 19.04 GB faltando.
- Pré-checagem de recursos com números reais desta máquina: disco 17.8 < 22.0 GB, RAM 1.7 < 13 GB → recusaria antes de baixar.
- Proveniência (pacotes, git commit/dirty, diferenças do `requirements.lock.txt`) em todo run.json.
- `benchmark generate/evaluate --resume` testados com interrupção simulada; retomada recusa ambiente/código/vídeo alterados.
- Métricas motion-v2 (trajetória, escala, cabeça, yaw do tronco) e página de revisão visual.
- Testes: **64 passed** (CPU). E2E com MediaPipe real: foto "astronaut" com deriva sintética → rotação relativa da
  cabeça 0.0° (fiel) vs 11.4° (deriva); trajetória null (quadris fora do quadro, correto).
- Continua sem geração real: sem CUDA, sem pesos, sem mídia do usuário.

## Estado atual — 2026-09-24

- Baseline: implementado, validado apenas com pipeline minúsculo aleatório; geração real BLOQUEADA.
- Extração MediaPipe: implementada e testada em CPU.
- Avaliação: integridade e métricas de movimento implementadas e testadas em CPU.
- Benchmark: protocolo de 20 vagas e ferramentas implementados; mídias reais ausentes, conjunto não congelado.
- Adapter, identidade treinável e treinamento: não iniciados; MVP e equivalência ao Kling não demonstrados.

## Entrega desta sessão

- Auditoria inicial: branch main, HEAD 6943f0f, árvore limpa; alterações anteriores preservadas.
- `evaluation/motion.py`: PCK/erro de corpo e mãos, cobertura, expressão e aceleração, sem interpolação de ausências.
- `evaluation/protocol.py`: manifesto, validação e lock SHA-256 de mídia/condições.
- `evaluation/benchmark.py`: re-extração, relatório por caso, registro automático e comparação estrita.
- `scripts/benchmark.py`: check, prepare, freeze, generate, evaluate e compare.
- `benchmark/v1.draft.yaml`: 20 vagas em dez categorias; templates de revisão visual e registro de falhas.
- ADR-007 e EXP-005 documentam decisões, limitações e próximos experimentos.

## Verificação executada

```bash
.venv/Scripts/python scripts/check_env.py --cuda-test
.venv/Scripts/python -m pytest -q -p no:cacheprovider --basetemp outputs/test-benchmark-release-20260924
.venv/Scripts/python scripts/benchmark.py check
```

- Ambiente: Python 3.12.10, PyTorch 2.14.0+cpu, CUDA False; RAM disponível 0.94/15.69 GB; disco C livre 16.66 GB.
- Testes: **51 passed in 29.42s**. Execução com acesso fora do sandbox às pastas temporárias, devido às restrições já documentadas.
- Testes cobrem erros conhecidos de keypoints, ausências, alinhamento, adulteração de dados/lock, integridade, extração real MediaPipe e repetição de avaliação facial.
- Retrato astronaut: corpo/rosto detectados, mas quadris insuficientemente visíveis; PCK corporal null é esperado. Comparação facial do vídeo consigo mesmo teve erro zero e foi repetida. Não é vídeo gerado por difusão.
- Orquestração generate testada com baseline simulado; geração CUDA real permanece não testada.
- Check: código de saída 2, **BLOCKED**, 20 casos, 140 pendências de arquivos/metadados. Relatório local: outputs/benchmark-preflight.json.
- A mensagem de decoder sobre moov ausente durante pytest vem do teste de arquivo inválido; suíte terminou com código 0.

## Bloqueios reais

- Máquina atual sem CUDA; precisa executar na RTX 3060.
- Pesos Wan não presentes no cache; download estimado anteriormente em 19.04 GB segue sem autorização e sem espaço suficiente nesta máquina.
- Imagens/vídeos e fontes/licenças não fornecidos; nenhum dataset novo baixado.

## Limitações

- ~~PCK normalizado não mede trajetória global~~ → motion-v2 mede trajetória/escala/rotação; expressão não mede identidade.
- Aceleração não mede flicker de textura. Não há score global nem promoção automática.
- Identidade, fluxo óptico e qualidade perceptual continuam pendentes (head pose: motion-v2).
- ~~Loader legado não fixa revisão HF~~ → revisão fixada (ADR-008); model_revision gravado por caso.
- Compatibilidade com 8 GB, qualidade, tempo de geração real e ganho sobre baseline não medidos.

## Próxima ação

Preencher as mídias/metadados conforme benchmark/README.md. Na máquina CUDA,
executar primeiro EXP-001, depois congelar o benchmark e gerar duas execuções
comparáveis; avaliar e revisar antes de iniciar adapters ou treinamento.
