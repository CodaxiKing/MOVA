# Training (planejado — nada implementado)

Treino só começa depois de: baseline executado (EXP-001), extração validada em vídeo real (EXP-002), representação escolhida (EXP-003).

## Estrutura prevista
```text
training/
├── train_motion_encoder.py   # estágio A: encoder de movimento (pode pré-treinar com objetivo auxiliar de reconstrução de pose)
├── train_adapter.py          # estágio B: encoder + projection + adapter com backbone congelado
├── losses.py
├── dataset.py                # lê vídeos + .pt de movimento + latentes VAE pré-computados
├── trainer.py                # Accelerate, bf16, grad checkpointing, grad accumulation, resume
└── config.yaml
```

## Orçamento de memória (INFERENCE, a medir)
- DiT 1.3B bf16 congelado ≈ 2.6 GB; sem estados de otimizador.
- Latentes do VAE e embeddings de texto **pré-computados** → VAE e UMT5 fora da GPU no treino.
- 256×256×17 frames → ~1 280 tokens: ativações pequenas com gradient checkpointing.
- Adapter ~20–80M params + AdamW (8-bit se necessário).
- Batch 1, gradient accumulation 8–16.

## Losses — por que cada uma (implementar só quando justificada)

| Loss | Por que é necessária | Quando entra | Risco |
|---|---|---|---|
| **Reconstruction (flow-matching MSE)** | É o objetivo de treino nativo do Wan: prever a velocidade `v = ε − x0` no espaço latente. Sem ela o adapter não aprende a gerar o frame-alvo. | Estágio B, desde o início | nenhum; é a base |
| **Regional weighting (mãos/rosto)** | Mãos e rosto ocupam poucos pixels → contribuem pouco para a MSE média; MimicMotion mostrou ganho ao amplificar a loss nessas regiões. Máscaras vêm dos nossos keypoints. | Estágio B, depois do baseline do adapter | peso alto demais gera artefatos |
| **Motion / pose loss** | A MSE não garante que a pose gerada bate com a de referência. Re-extrair pose do vídeo gerado exige decodificar (caro). Alternativa barata: loss em features latentes preditas vs pose. | Só se a precisão de pose (métrica) ficar ruim | decodificar em 8 GB é caro; possivelmente só em validação |
| **Identity loss** | Referência deve ser preservada; face embedding (ArcFace-like) entre referência e frames gerados. Exige decodificar. | Fase de identidade | licença de pesos de face recognition (InsightFace é não comercial) |
| **Temporal consistency** | Reduz flicker. Como o backbone já é um modelo de vídeo, pode ser desnecessária; medir primeiro com métrica temporal. | Só se a métrica temporal piorar vs baseline | pode suavizar movimento real |
| **Face consistency** | Subcaso de identidade restrito ao crop do rosto. | Junto com identity | idem |
| **Hand/pose keypoint loss** | Erro de keypoints re-extraídos das mãos. | Depois de regional weighting, se ainda ruim | custo de decodificação |

Princípio: começar só com flow-matching MSE (+ ponderação regional), medir com `evaluation/`, e adicionar losses quando uma métrica específica indicar o problema.

## Estágios de treino (inspirados em evidência pública)
1. Corpo apenas (Motion Encoder de corpo + adapter), poucos frames.
2. Mais frames (consistência temporal).
3. Face e mãos (encoders extras, tokens via cross-attention).
4. Identity Encoder.
(Evidência: treino em 2 estágios do Animate Anyone/Moore; "progressive multi-stage training" do Kling tech report.)
