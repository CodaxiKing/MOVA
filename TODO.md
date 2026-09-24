# TODO

## Critical

- [ ] Usuário: autorizar download de `Wan-AI/Wan2.1-VACE-1.3B-diffusers` (19.04 GB, Apache 2.0) na máquina da RTX 3060
- [ ] Usuário: fornecer `assets/reference/maya.png` e `assets/motion/dance.mp4`
- [ ] Na máquina RTX 3060: `check_env.py --cuda-test` e registrar VRAM/RAM/disco reais em `CLAUDE.md`
- [ ] EXP-001: executar baseline 256px/17 frames, medir VRAM pico e tempo, registrar

## Qualidade do motion control (ADR-011..014) — próximos passos

- [x] identity-v1, temporal-v1, GSB cego; integrados ao evaluate/compare/review
- [x] Retargeting por comprimento de ossos; One-Euro; pós-processamento de mãos; controle a partir das tracks
- [x] Encoders, Identity Encoder v0, fusão, Motion Adapter zero-init; scaffolding de treino; `mova train --smoke`
- [x] Auditoria de licenças; registro de fontes; validação de pares; build do manifesto; `benchmark intake`
- [ ] Calibrar limiares identity/temporal (`REVIEW_HINTS`) com vídeos gerados revisados
- [ ] Autorizar DINOv2-small (88 MB, Apache-2.0) para o embedding de identidade, se desejado
- [ ] Primeira sessão GSB real (baseline VACE vs baseline + retarget) quando houver geração na GPU
- [ ] Testar retargeting com personagem real de corpo inteiro (torso visível) e com o baseline VACE
- [ ] EXP-003 com tracks reais; decidir trocar a entrada do corpo para rot6d dos ossos longos
- [ ] EXP-006 com vídeo real: decidir suavização e medir o efeito do pós-processamento de mãos
- [ ] Na GPU: `mova train` com o transformer real (UNVERIFIED), medir VRAM do adapter em 8 GB
- [ ] Integrar o condicionamento MOVA treinado no `mova infer` (novo modelo no registry com checkpoint avaliado)
- [ ] Auditar TikTok dataset, UBC Fashion, Champ; registrar licença Pexels clipe a clipe ao usar HumanVid

## Arquitetura (ADR-009/010) — próximos passos

- [x] Runtime (PyTorchRuntime), DeviceManager, PrecisionManager, MemoryManager, erros estruturados
- [x] Interface de modelo, ModelSpec, registry, Wan VACE + modelo minúsculo de smoke test
- [x] Core: config (`configs/runtime.yaml`), capabilities, serviço de inferência; CLI `mova`
- [x] Regressão CPU bit-exata contra o baseline da Fase 0
- [ ] Na RTX 3060: `mova info` (validar TorchDeviceProbe: nome, VRAM, cc, bf16), `mova infer --model tiny --device cuda`
      (valida place/offload/VRAM stats na GPU), depois EXP-001 com `mova infer --model wan`
- [ ] Medir VRAM antes/pico/depois e tempo por precisão (bf16/fp16) na GPU; atualizar `ModelSpec.verification`
- [ ] Fase 11 API: decidir dependência (FastAPI + uvicorn: licenças MIT/BSD) e criar `api/` chamando `core.inference`
- [ ] Fase 9: definir interfaces de encoders (identidade/rosto/mãos/temporal) junto da 1ª implementação real (Fase 3 do roadmap)
- [ ] Fase 18: config de produção reproduzível (modelo+config+runtime+hardware+seed+input)
- [x] `mova train` quando existir treino (usar runtime/ para device/precisão)
- [ ] ONNX/TensorRT (Fases 13/14) só após export validado + benchmark; multi-GPU (15) só com hardware; nativo (16) só após profiling
- [ ] Testar Linux e AMD/ROCm quando houver ambiente (hoje UNVERIFIED)
- [ ] `ExperimentRun` usa run_id por segundo: duas execuções no mesmo segundo compartilham pasta (existente, não corrigido)

## High Priority

- [x] Protocolo de benchmark com 20 vagas, check, preparo e lock por SHA-256
- [x] Avaliação CPU de corpo/mãos/expressão/aceleração com cobertura e dados ausentes
- [x] Comparação estrita de relatórios e modelos de revisão/falhas (sem promoção automática)
- [ ] Preencher as 20 vagas com mídias autorizadas, revisar recortes e congelar benchmark real
- [ ] Executar duas gerações equivalentes na GPU, medir e revisar antes de treinar

- [x] Validação automática da integridade de output.mp4 (requisito 49.18), testada em CPU

- [x] Pesquisa Fase 0 (`docs/research/`)
- [x] Seleção de backbone (ADR-001/002)
- [x] `scripts/inference_baseline.py` com gates de download/CPU
- [x] Extração separada body/face/hands com prévias
- [x] Registro automático de experimentos
- [ ] EXP-002: extração em vídeo real de dança (taxa de detecção, jitter, mãos)
- [ ] EXP-004: pose MediaPipe→OpenPose vs DWPose como entrada do VACE
- [ ] Recalibrar thresholds de `select_profile` com medições reais

- [x] Fixar revisão exata do modelo e versões das dependências (ADR-008, `requirements.lock.txt`)
- [x] Verificar cache completo (17 arquivos, tamanho + hash) em vez de só `model_index.json`
- [x] Pré-checagem de RAM/disco/VRAM antes de iniciar
- [x] Retomar `benchmark generate` e `evaluate` interrompidos
- [x] Medir trajetória global, escala, rotação da cabeça e yaw do tronco (motion-v2)
- [x] Relatório visual por caso (vídeo lado a lado, métricas, alertas, review.csv)
- [ ] Recalibrar estimativas de RAM/VRAM de `common/resources.py` com EXP-001
- [ ] Calibrar limiares de alerta (`evaluation/review.py::REVIEW_HINTS`) com casos revisados

## Medium Priority

- [ ] EXP-003: comparar representações (2D, 3D world, normalizada, velocidade, rot6d) — reconstrução/predição e jitter
- [x] Retargeting de proporções (comprimento de ossos) do ator para a personagem
- [ ] Extrator DWPose ONNX opcional (Apache 2.0; ~2 arquivos ONNX, pedir autorização se >50 MB)
- [ ] Baseline B: Wan2.1-Fun-V1.1-1.3B-Control (19.81 GB — autorização)
- [ ] `evaluation/`: métricas de pose (PCK/erro de keypoints re-extraídos), identidade (face embedding), temporal (warp/flow error), qualidade (FVD/CLIP)
- [x] Parte inicial de evaluation: PCK/erro, mãos separadas, blendshape MAE, aceleração; demais métricas acima pendentes
- [x] Protótipo do Motion Adapter (encoder + projection + temporal attention, zero-init) com testes de forma em CPU

## Low Priority

- [x] Suavização One-Euro para keypoints
- [ ] Vídeos longos: janelas sobrepostas (progressive latent fusion)
- [ ] Integração ComfyUI (opcional)

## Research

- [ ] Descobrir o que é "Motion Mirror" (pedir link ao usuário)
- [ ] Prompt Enhancer (PE) do Kling: sem implementação pública — decidir se o MOVA precisa de controle semântico por texto no MVP (provavelmente não; baseline VACE já aceita prompt)
- [x] Avaliação GSB pareada humana por eixo (5 eixos do Kling) como extensão de `evaluation/review.py` / review.csv
- [x] Auditar licença dos pesos LivePortrait / InsightFace antes de usar como face encoder
- [x] Verificar licença do UniAnimate-DiT
- [x] Reverificar termos do AIST++ (factsheet deu 404)
- [ ] Ler o paper completo do Wan-Animate-2 (arquitetura do encoder de vídeo de condução)
- [ ] Ler seções de método do Wan-Animate (2509.14055) sobre face adapter

## Experiments

- [ ] EXP-001 baseline VACE-1.3B
- [ ] EXP-002 extração em vídeo real
- [ ] EXP-003 representação de movimento
- [ ] EXP-004 DWPose vs MediaPipe render

## Technical Debt

- [ ] Fixar revisão HF no loader baseline e identificar checkpoint nos registros; atualmente model_revision=null
- [ ] Medir trajetórias globais e head pose; métricas normalizadas atuais removem translação/escala

- [ ] `select_profile` thresholds são chutes iniciais
- [ ] `encode_prompts_cached` replica a lógica de padding do Diffusers (sem `prompt_clean`) — validar igualdade numérica contra `pipe.encode_prompt` quando os pesos existirem
- [ ] Mapeamento de face no render OpenPose usa pontos do mesh MediaPipe (subamostrados), não os 68 do DWPose
