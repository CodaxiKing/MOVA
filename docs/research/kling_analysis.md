# Kling Motion Control — Análise Pública

> **Aviso:** o MOVA **não** reproduz, copia ou reconstrói a arquitetura, o código ou os pesos proprietários da Kuaishou.
> Este documento separa estritamente o que foi publicado do que é inferência nossa.
> Última verificação: 2026-09-24.

## Fontes

| Fonte | Tipo | Link |
|---|---|---|
| Kling-MotionControl Technical Report | Technical report (arXiv 2603.03160, mar/2026) | https://arxiv.org/abs/2603.03160 |
| HF Papers page | Índice | https://huggingface.co/papers/2603.03160 |
| Kling v3 Motion Control (API via Replicate, publicada pela conta `kwaivgi`) | Documentação de uso | https://replicate.com/kwaivgi/kling-v3-motion-control |
| Press release Kling 3.0 (set/2026) | Marketing — **não** tratado como fonte técnica | https://www.openpr.com/news/4630442/kling-ai-launches-kling-3-0-kuaishou-s-ai-director-platform |
| Wan-Animate-2 paper (comparação com KLING MotionControl) | Paper de terceiros | https://arxiv.org/abs/2608.06009 |

"Element Binding" e "Kling Motion Control 3.0": não encontramos documentação técnica primária específica além do technical report e das páginas de API. Tratamos detalhes sobre eles como **UNKNOWN**.

---

## 1. Publicly documented (FACT — technical report 2603.03160)

- **Backbone:** "unified DiT-based framework" para animação holística de personagens, sobre um DiT com compressão por **3D VAE**. **Tamanho do modelo não divulgado.**
- **Movimento "divide-and-conquer":** representações de movimento **heterogêneas por região** (corpo, rosto, mãos), orquestradas por treinamento **progressivo multi-estágio**.
- **3D awareness:** representações multi-granulares "endowed with 3D perception capabilities through large-scale multi-view supervision" — alinhamento entre orientações diferentes do personagem e controle de câmera por texto.
- **Identity-agnostic learning:** abstração geométrica que desacopla "dynamic patterns from the driving subject's physical attributes" + modelagem semântica do movimento ("high-level intent") → retargeting entre morfologias diferentes.
- **Identidade:** "dedicated identity encoding and fusion mechanism" + **subject library** que aceita imagens multi-view ou clipes de vídeo do personagem.
- **Prompt Enhancer (PE):** módulo que concilia controle de movimento e texto (cenário, roupa, câmera).
- **Dados:** dataset massivo com vários tipos de personagem, incluindo **dados renderizados** e **câmeras de alta velocidade**; filtragem por qualidade, dinâmica de movimento, consistência do sujeito; anotações de ações, micro-expressões, interação humano-objeto e movimentos de câmera.
- **Aceleração:** distilação multi-estágio (~10× speedup); "dual-branch sampling" para CFG multi-condição; "gradient merging" para evitar custo de CFG.
- **Avaliação:** GSB (Good/Same/Bad) humano, métrica (G+S)/(B+S), 150 casos, dimensões: Visual Quality, Dynamic Quality, Identity Preservation, Motion Accuracy, Expression Accuracy. Comparado a Dreamina, Runway Act-Two, Wan-Animate. Resolução de avaliação 1080P.

## 2. Não divulgado (UNKNOWN)

- Qual DiT/checkpoint base, número de parâmetros, camadas.
- Arquitetura exata dos extratores/encoders de movimento (keypoints? SMPL? features latentes?).
- Mecanismo exato de fusão de identidade (cross-attention? concatenação temporal? tokens?).
- Funções de perda, hiperparâmetros, estágios de treino, tamanho do dataset, compute.
- Requisitos de hardware de inferência. Ablations. Casos de falha.
- Detalhes internos de "Element Binding".

## 3. Engineering inference (INFERENCE — nossas deduções, não confirmadas)

| # | Inferência | Base pública | Como usamos no MOVA |
|---|---|---|---|
| I1 | Corpo, rosto e mãos têm encoders/representações separados antes da fusão | "heterogeneous representations tailored to each body region" | Pipeline de extração separado `body/face/hands` (Fase 2) |
| I2 | A representação de corpo carrega informação 3D (profundidade/orientação), não só keypoints 2D | "3D perception... multi-view supervision" | Testar 2D vs 3D (MediaPipe world landmarks) no EXP de representação |
| I3 | Normalização da geometria (proporções do corpo) é feita antes de condicionar o gerador | "geometric abstraction decoupling... physical attributes" | Normalizar keypoints por escala do torso / retargeting de comprimento de ossos |
| I4 | Identidade entra por um caminho distinto do movimento, com suporte a múltiplas vistas | "identity encoding and fusion" + "subject library" | Identity Encoder aceitando N imagens (Fase de identidade) |
| I5 | Treino em estágios: corpo primeiro, depois rosto/mãos | "progressive multi-stage training" | Treinar Motion Adapter de corpo antes de face/hands |

## 4. Hipóteses que NÃO assumimos

- Que o Kling usa o mesmo esquema de injeção do Wan-Animate.
- Que o Kling usa DWPose/SMPL/qualquer extrator específico.
- Qualquer número de parâmetros ou resolução interna.

## 5. Conclusão para o MOVA

Os princípios publicados que adotamos como **direção de pesquisa** (não como cópia): (a) separação por região do movimento, (b) desacoplar geometria do ator do movimento, (c) identidade por caminho próprio com múltiplas vistas, (d) treino progressivo, (e) avaliação em eixos separados (identidade, movimento, expressão, qualidade dinâmica). Nossa implementação concreta é baseada em componentes open-source documentados (Wan2.1, VACE, DWPose/MediaPipe) — ver `backbone_selection.md` e `docs/architecture.md`.
