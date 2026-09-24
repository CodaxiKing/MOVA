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
| Guia oficial do produto Motion Control | **Docs oficiais (Kling)** | https://kling.ai/quickstart/motion-control-user-guide |
| Feature page / FAQ (2.6 vs 3.0) | Marketing oficial (FAQ) | https://kling.ai/feature/ai-motion-control |
| Guia do modelo base Video 3.0 | Docs oficiais | https://kling.ai/quickstart/klingai-video-3-model-user-guide |
| API schema (parâmetros oficiais) | Docs de API via Replicate (`kwaivgi`) | https://replicate.com/kwaivgi/kling-v3-motion-control/api |

Páginas JS-only inacessíveis por fetch (marcar **UNKNOWN** se precisarmos delas): release note `kling.ai/release-note/release-notes/4titicw2vg`, app `app.klingai.com/global/video-motion-control/new`, API docs `kling.ai/document-api/api/video/motion-control`.
Sites afiliados/SEO (klingmotioncontrol.com, kling3pro.com, motioncontrolai.io etc.) **não são fonte** — termos como "Motion Brush"/"skeletal anchoring" só aparecem lá e são descartados.

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
- **Avaliação:** GSB (Good/Same/Bad) humano, métrica (G+S)/(B+S), 150 casos, dimensões: Visual Quality, Dynamic Quality, Identity Preservation, Motion Accuracy, Expression Accuracy. Comparado a Dreamina, Runway Act-Two, Wan-Animate. Resolução de avaliação 1080P. **Sem métricas objetivas** (FID/FVD): o paper diz que "incorporará métricas objetivas no futuro". **Números GSB Overall (Tab. 1):** vs Dreamina 3.44; vs Runway Act-Two 16.25; vs Wan-Animate 4.00 — são **preferência humana reportada pelo próprio autor**, sem ablations, não evidência de superioridade mensurável (INFERENCE crítica).
- **Estrutura do paper (FACT):** Abstract / Introduction / Evaluation / Related Work / Conclusion — **não há seção Method**, nem equações, apêndice, ablations, parâmetros ou detalhes de treino.

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

---

## 6. Produto — "Kling VIDEO 3.0 Motion Control" (verificado 2026-09-24)

> O **paper não usa o termo "3.0"** (0 ocorrências em 2603.03160). "3.0" é a versão do **produto**; se é o mesmo modelo do paper é **INFERENCE** (datas compatíveis: paper 05/03, guia 05/03, modelo Replicate 05/03 — mas nenhuma fonte afirma equivalência).

### FACT (docs oficiais Kling + schema Replicate)

- **O que é novo no 3.0 vs 2.6:** "significantly improves **facial consistency** across complex motion, multiple angles, and longer sequences" (FAQ oficial); "improved element consistency and better identity preservation" (Replicate). Motion Control já existia no VIDEO 2.6.
- **Element Binding** (nome de produto do "identity encoding/fusion" — correspondência = INFERENCE): cria um *element* a partir de 2–4 imagens ou vídeo curto; a **Element Library usa apenas informação facial** (não roupa/cabelo/maquiagem/props); funciona só quando a orientação do personagem casa com a do vídeo (`character_orientation=video`).
- **Entradas:** reference image (340–3850 px, ≤10 MB, aspecto 1:2.5–2.5:1) + driving video (`.mp4/.mov`, 3–30 s, ≤100 MB; `orientation=image` → 3–10 s). Prompt de texto **opcional** (cena/roupa/câmera). `keep_original_sound` (default true).
- **Requisitos do vídeo de referência:** plano-sequência contínuo, sem cortes nem movimentos de câmera, **sujeito único**, personagem sempre visível.
- **Saída:** 720p (`std`) / 1080p (`pro`, default); duração acompanha o upload (mínimo 3 s de ação válida).
- **Preço oficial (guia):** 3.0 MC → 12 credits/s (Pro), 9 (Std). Página dev → US$ 0,168/s (1080p), US$ 0,126/s (720p). Replicate → US$ 0,12/s (pro), US$ 0,07/s (std). *Terceiros (Apiframe) citam 5/8 credits = preço do 2.6, provavelmente desatualizado.*
- **Velocidade:** 1 amostra pública no Replicate: 13,4 s de vídeo → **~521 s** de geração (ordem de vários minutos; single data point, não SLA). Claim do paper: >10× com destilação.
- **Limites declarados:** sujeito único; degrade se o rosto do elemento divergir do primeiro frame; degradação com tipos de corpo muito diferentes; **não** é para lip-sync (usar Video 3.0 padrão).
- **Contexto Kling 3.0 (modelo base):** Multi-Shot, Element Consistency, áudio nativo, 3–15 s flexível; camera movement via prompt no modo `orientation=image`.

### Terceira descrição independente (FACT, Wan-Animate-2 2608.06009 §5)

- User study cego: Kling-MotionControl = "**comparable performance**" vs Wan-Animate-2 (maioria avaliou qualidade equivalente).
- Observação deles: "Dreamina e **Kling-MotionControl são construídos sobre modelos foundation maiores e fechados**"; Wan-Animate-2 é open-source.
- Crítica qualitativa agrupada das concorrentes: "struggle with **expression fidelity, detailed hand articulation, and body shape misalignment**" (agrupada, **não** atribuída nominalmente ao Kling).

### O que continua UNKNOWN

Representação concreta de movimento por região; mecanismo de injeção no DiT (0 ocorrências de "cross-attention"/"adapter" no paper); nº de parâmetros; dataset (tamanho/anotações); hiperparâmetros; NFE do student; fps; equivalência paper ↔ produto.
