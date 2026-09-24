# Wan Family — Análise Técnica

Última verificação: 2026-09-24. Fontes em `sources.md`.

## Wan2.1 (base)

**FACT**
- DiT + flow matching; Wan-VAE 3D causal (compressão 4× temporal, 8× espacial); patch 1×2×2 → num_frames deve ser `4k+1` e H/W múltiplos de 16.
- 1.3B: dim 1536, 12 heads, 30 blocos, FFN 8960. Texto via UMT5-XXL (cross-attention).
- T2V-1.3B a 480P: 8.19 GB em RTX 4090 com offload do modelo e T5 em CPU.
- Apache 2.0.

**INFERENCE**
- Em 256×256 e 9–17 frames a sequência de tokens do DiT é pequena: (17−1)/4+1 = 5 latentes temporais × (256/16)² = 5×256 = **1 280 tokens** (vs ~32 760 tokens em 480P/81 frames). O custo de atenção cai ~650×. Isso torna a Fase de adapter viável em 8 GB.
- O maior consumidor de memória em 8 GB não é o DiT 1.3B (~2.6 GB em bf16), mas o **UMT5-XXL (~11 GB em bf16)** → precisa ficar em CPU ou ser pré-computado e cacheado (embeddings de prompt fixos). Isso é uma decisão de engenharia importante (ADR-004).

## Wan2.1-VACE-1.3B

**FACT**
- Adiciona "VACE layers" (blocos de contexto) que recebem latentes de controle (vídeo de controle + máscara + imagens de referência) e somam seus resultados em blocos selecionados do DiT, com `conditioning_scale` por camada.
- Diffusers: `WanVACEPipeline(video, mask, reference_images, conditioning_scale, height, width, num_frames, ...)`.
- Suporta pose-to-video + reference images simultaneamente ("composition").

**INFERENCE**
- É exatamente o padrão "Motion Adapter aditivo sobre backbone congelado" que queremos construir: blocos paralelos ao backbone que somam residuais. Nosso adapter pode seguir o mesmo contrato (residual zero-init por bloco), o que facilita comparação justa com o baseline.

## Wan2.2-Animate-14B

**FACT (paper 2509.14055 + Diffusers)**
- Esqueleto espacialmente alinhado (pose video) para corpo.
- Features faciais implícitas de crops do rosto (face video) para expressão.
- Paradigma de entrada modificado: referência vs regiões a gerar numa representação simbólica comum (animation e replacement).
- Relighting LoRA (replacement).
- Geração por segmentos de 77 frames com 1 frame de condicionamento do segmento anterior.
- Sem código de treino.

**INFERENCE**
- Separar corpo (sinal espacial denso, alinhado) de rosto (sinal compacto, implícito) é justamente a divisão "Body Motion Encoder" vs "Face Motion Encoder" da nossa arquitetura hipotética — há evidência open-source de que funciona.

## Wan-Animate-2

**FACT (paper 2608.06009)**
- Consome o vídeo de condução diretamente, **sem extratores intermediários**.
- Controle de viewpoint via texto; variante Lite real-time (Self-Forcing distillation).
- 14B, 8×A800 para 720P, sem código de treino.

**INFERENCE**
- Extratores explícitos (pose) limitam fidelidade (erros do extrator, perda de nuance). A tendência SOTA é motion features latentes aprendidas. Para o MOVA isso sugere um roteiro: começar com keypoints explícitos (interpretáveis, baratos, depuráveis) e **depois** experimentar um encoder latente de movimento treinado com os keypoints como supervisão auxiliar.

## Resumo de viabilidade em RTX 3060 8 GB

| Modelo | Inferência 8 GB | Treino adapter 8 GB | Nota |
|---|---|---|---|
| Wan2.1-T2V-1.3B | Sim (oficial ~8.2 GB c/ offload) | Provável (DiffSynth ~6 GB, não verificado) | backbone |
| Wan2.1-VACE-1.3B | Provável c/ offload + baixa res | — (baseline) | **baseline** |
| Wan2.1-Fun-V1.1-1.3B-Control | Provável c/ offload/qfloat8 | Provável | baseline B |
| Wan2.2-Animate-14B | Só GGUF/block swap, muito lento | Não | referência |
| Wan-Animate-2 14B | Não prático | Não | referência |
