# Backbone Selection

Data: 2026-09-24 · Status: **Decidido (ADR-001, ADR-002)** · Nenhum número de VRAM aqui foi medido por nós ainda.

## Decisão

| Papel | Modelo | Por quê |
|---|---|---|
| **Backbone de desenvolvimento** (congelado, recebe nossos adapters) | **Wan2.1 DiT 1.3B** (pesos do `Wan2.1-VACE-1.3B` / `Wan2.1-T2V-1.3B`) | DiT de vídeo pequeno, Apache 2.0, treinável com LoRA em GPU consumer |
| **Baseline da Fase 1** | **Wan2.1-VACE-1.3B via Diffusers `WanVACEPipeline`** (pose video + reference image) | Faz exatamente reference+motion → vídeo, sem treino, no mesmo backbone que vamos adaptar |
| Baseline B (opcional) | Wan2.1-Fun-V1.1-1.3B-Control (VideoX-Fun) | Mesmo backbone, outro mecanismo de controle (canais concatenados) — comparação |
| Arquitetura de referência (não executada localmente) | Wan2.2-Animate-14B, Wan-Animate-2 | Evidência open-source da separação corpo/rosto/identidade; grandes demais |
| Baseline U-Net (opcional) | MimicMotion | Comparação com família diferente |

## Critérios e pontuação

Escala 0–3 (3 = melhor). "Treino" = código/ecossistema de fine-tuning disponível.

| Critério | Wan2.1-VACE-1.3B | Wan-Fun-1.3B-Ctrl | Wan2.2-Animate-14B | Wan-Animate-2 | MimicMotion | Moore-AA |
|---|---|---|---|---|---|---|
| Qualidade / motion transfer | 2 | 2 | 3 | 3 | 2 | 1 |
| Arquitetura DiT (meta final) | 3 | 3 | 3 | 3 | 0 | 0 |
| Código disponível | 3 | 3 | 2 | 2 | 2 | 3 |
| Pesos disponíveis | 3 | 3 | 3 | 3 | 2 (gated) | 3 |
| Licença | 3 Apache | 3 Apache | 3 Apache | 3 Apache | 1 SVD Community | 2 (SD1.5 OpenRAIL) |
| Modificável / adapters | 3 | 3 | 2 | 1 | 1 | 2 |
| Treino | 2 (DiffSynth) | 3 (VideoX-Fun) | 0 | 0 | 0 | 3 |
| Memória em 8 GB | 2 | 2 | 0 | 0 | 2 | 1 |
| Comunidade / docs | 3 | 2 | 3 | 2 | 2 | 1 |
| Separar motion × identity | 2 (controle vs refs) | 2 | 3 | 2 | 1 | 2 |
| **Total** | **26** | **26** | 22 | 19 | 13 | 18 |

Empate VACE × Fun: VACE vence o papel de *baseline* porque (1) está nativamente no Diffusers com `enable_model_cpu_offload`, group offloading e VAE tiling — menos código próprio e menos dependências; (2) seu mecanismo (blocos de contexto que somam residuais em camadas do DiT) é o mesmo contrato que nosso Motion Adapter vai seguir, o que torna a comparação BASELINE × NOSSO direta. Fun-Control fica como baseline B e como referência de código de treino.

## Justificativa técnica

1. **Não é o "mais poderoso", é o maior que cabe e treina.** Wan-Animate-2/Animate-14B são SOTA mas: 14B+ params, sem código de treino, dezenas de GB de VRAM. Em 8 GB só rodariam por GGUF + block swap, muito lento, e nunca treinariam.
2. **Custo de tokens escala bem para baixo.** Em 256×256 × 17 frames o DiT vê ~1 280 tokens (vs ~32 760 em 480P×81), ver `wan_analysis.md`. Isso é o que torna treinar adapters viável.
3. **O gargalo de memória é o encoder de texto, não o DiT.** Tamanhos reais (HF API, 2026-09-24):

   | Repo | text_encoder | transformer | vae | total |
   |---|---|---|---|---|
   | Wan-AI/Wan2.1-VACE-1.3B-diffusers | 11.36 GB | 7.15 GB | 0.51 GB | **19.04 GB** |
   | Wan-AI/Wan2.1-T2V-1.3B-Diffusers | 22.72 GB (fp32) | 5.68 GB | 0.51 GB | 28.94 GB |
   | alibaba-pai/Wan2.1-Fun-V1.1-1.3B-Control | — | — | — | 19.81 GB |

   Estratégia: UMT5 roda em CPU uma vez, embeddings do prompt são cacheados em disco (ADR-004). No treino o encoder de texto nem é carregado.
4. **Licença limpa.** Toda a cadeia (Wan2.1, VACE, Diffusers, DiffSynth, MediaPipe, DWPose) é Apache 2.0.
5. **Caminho de upgrade.** Se o 1.3B for o limite de qualidade, os adapters são reprojetados para Wan2.2-TI2V-5B ou 14B (mesma família, mesmas convenções de VAE/patch) — ver ADR futuro.

## Requisitos para executar o baseline (aguardando autorização do usuário)

| Item | Valor |
|---|---|
| Download | **19.04 GB** (`Wan-AI/Wan2.1-VACE-1.3B-diffusers`) |
| Espaço em disco recomendado | ≥ 25 GB livres (download + cache + saídas) |
| Licença | Apache 2.0 |
| VRAM estimada | ≤ 8 GB com `enable_model_cpu_offload` + VAE tiling a 256–480P (INFERENCE; base oficial T2V-1.3B = 8.19 GB a 480P) |
| RAM do sistema | ≥ 16 GB (UMT5 bf16 ~11 GB é carregado em CPU) — **ponto de risco**; mitigação: embeddings cacheados |
| Por que é necessário | É o baseline e contém o backbone que receberá nossos adapters — o mesmo download serve às Fases 1 e 3 |

## Riscos

- **R1 — RAM do sistema:** 16 GB de RAM com UMT5 (11 GB) + transformer em CPU durante offload pode estourar. Mitigação: carregar text encoder isolado, cachear embeddings, liberar, depois carregar o resto.
- **R2 — Qualidade do 1.3B** em mãos/rosto é limitada; esperado. É o que vamos medir e tentar melhorar.
- **R3 — Formato do pose video:** VACE foi treinado com poses estilo OpenPose/DWPose. Nosso render a partir de MediaPipe é aproximação; testar ambos (EXP planejado).
