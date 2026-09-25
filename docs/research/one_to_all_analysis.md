# One-to-All Animation e lições externas para as falhas medidas (2026-09-24)

Motivo: EXP-001 mediu corpo PCK 0.78, **mãos PCK 0.07**, rosto detectado em 73 % dos quadros e rosto "não
idêntico" (queixa do usuário). Esta nota cruza trabalhos públicos com essas falhas. FACT = fonte primária lida;
INFERENCE = dedução nossa.

## One-to-All Animation (Shi et al., arXiv 2511.22940)

Fontes: https://arxiv.org/abs/2511.22940 · https://github.com/ssj9596/One-to-All-Animation ·
https://huggingface.co/MochunniaN1/One-to-All-1.3b_1 (conta confirmada pelo README oficial, que aponta para
`MochunniaN1/One-to-All-sub`).

**FACT**
- Base: `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` (mesma família/tamanho do nosso backbone). Checkpoints 1.3b_1 (melhor em
  vídeo), 1.3b_2 (câmera em movimento), 14b.
- Licença: código Apache-2.0 (repo); pesos `license:apache-2.0` (card HF). 1.3b_1 = 6.57 GB, revisão 99dc3796f33b.
- Treino liberado: 3 estágios progressivos a partir do T2V-1.3B — (1) Reference Extractor, (2) Pose Control,
  (3) Token Replace (vídeos longos). Subconjunto de dados de treino publicado (`MochunniaN1/One-to-All-sub`; licença
  ainda NÃO auditada).
- "Alignment-free": treino reformulado como outpainting auto-supervisionado que leva referências de layout
  diferente a um formato comum; controle de pose "identity-robust" que desacopla aparência do esqueleto.
- Pose: DWPose; o README diz que cores mais claras no esqueleto/landmarks do rosto melhoram a consistência de
  identidade.
- Roda em T4 16 GB (Kaggle, via ComfyUI, segundo o README). Nenhum número para 8 GB.

**Verificado localmente (sem download):** VAE e tokenizer do T2V-1.3B têm o mesmo SHA-256 dos do VACE-1.3B já em
cache. O text encoder difere só em precisão/sharding (T2V fp32 22.7 GB, VACE bf16) — INFERENCE: mesmos pesos UMT5.

**INFERENCE para o MOVA:** é, na prática, o "Identity Encoder + adapter treinado" do caminho C, já treinado pelos
autores em dados grandes, sobre a nossa família de backbone e com licença permissiva. Em vez de treinar do zero,
o caminho mais curto é (a) avaliá-lo contra o baseline VACE com as mesmas métricas e (b) se for melhor, ajustá-lo
(fine-tune) com o código de treino deles.

## StableAnimator++ (arXiv 2507.15064)

**FACT (abstract):** distorção facial e perda de identidade aparecem quando referência e driver diferem em tamanho/
posição do corpo; propõe alinhamento de pose aprendido (matriz de similaridade guiada por SVD), Face Encoder
"content-aware" + ID Adapter "distribution-aware" contra a interferência das camadas temporais, e otimização
facial baseada em HJB durante o denoising. Backbone e licença dos pesos não informados no abstract; a versão 1 usa
SVD + ArcFace/InsightFace (pesos não comerciais — **bloqueado** no nosso registro).

## Mapeamento falha → técnica pública

| Falha medida (EXP-001) | Técnica | Fonte | Custo para nós |
|---|---|---|---|
| Rosto não idêntico / deformado | Encoder de referência dedicado (Reference Extractor) | One-to-All | pronto (pesos Apache-2.0) |
| Rosto pequeno (~6 tokens) | Features faciais implícitas de crops do rosto, fora do grid de latentes | Wan-Animate (2509.14055) | alto (14B; ideia reaproveitável) |
| Referência × driver com enquadramento diferente (corpo inteiro × coxas para cima) | Treino alignment-free por outpainting; alinhamento de pose aprendido | One-to-All; StableAnimator++ | pronto no One-to-All |
| Mãos PCK 0.07 | DWPose (mãos mais precisas) + amplificação regional da loss nas mãos | MimicMotion; One-to-All usa DWPose | médio (EXP-004) |
| Vídeo curto (17–49 quadros) | Token replace / segmentos com quadro de condicionamento | One-to-All; Wan-Animate (77 + 1) | pronto no One-to-All |

## Não usar
- Pesos de modelos fechados (Kling, Runway) ou saídas deles como dados de treino (termos de uso).
- Encoders InsightFace/ArcFace (pesos não comerciais), já bloqueados em `licenses.md`.
