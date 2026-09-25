# Auditoria de licenças — dados e pesos (2026-09-24)

Fontes primárias consultadas nesta data (repositórios oficiais, API do GitHub, API do Hugging Face, página de
termos). Estado resumido em `datasets/registry.yaml`, que é o que `scripts/datasets.py` aplica. Não é parecer
jurídico: registra o que as fontes dizem e as decisões de engenharia derivadas.

| Item | O que a fonte diz (FACT) | Decisão MOVA |
|---|---|---|
| **InsightFace** (código + modelos buffalo_l/antelopev2) | Código MIT. "The training data containing the annotation (and the models trained with these data) are available for non-commercial research purposes only." | **Bloqueado**. A métrica de identidade usa geometria MediaPipe (Apache-2.0) + DINOv2 opcional. |
| **LivePortrait** | `LICENSE` = MIT (Kuaishou). O repositório HF `KlingTeam/LivePortrait` (tag `license:mit`) inclui `insightface/models/buffalo_l/det_10g.onnx` e `2d106det.onnx`. | **Bloqueado** como pacote: os detectores InsightFace embutidos herdam a restrição não comercial. Um fork com MediaPipe (ComfyUI-LivePortraitKJ) precisaria de auditoria própria. |
| **UniAnimate-DiT** | GitHub API: `license: null`; sem arquivo `LICENSE` (404). README: "intended for academic research". | **Bloqueado**: sem licença explícita, todos os direitos são reservados. |
| **AIST++ / AIST Dance DB** | Anotações CC BY 4.0 (Google, código Apache-2.0). Vídeos: "may not be used for any purpose other than academic research… Use for commercial purposes is not permitted without prior written consent from AIST… Unauthorized redistribution… is prohibited." Exige formulário. | **research_only**. Aceito só em manifestos `intended_use: research`. A factsheet antiga (404) foi substituída por estes termos. |
| **HumanVid (reais, Pexels)** | Listas de URLs CC-BY-4.0; vídeos sob a licença e os **Termos de Serviço do Pexels**, que proíbem "scraping and the use of programs or robots for automatic data collection [...] including without limitation for machine learning purposes" e "bulk, large-scale or systematic copying" sem permissão explícita (reauditado 2026-09-24). | **blocked** — só com permissão escrita do Pexels. |
| **HumanVid (sintéticos, Unreal)** | Hub `zhenzhiwang/HumanVid` @ 12ecc587: card `license:apache-2.0`; README do GitHub CC-BY-4.0. 201.7 GB, ~50 000 clipes. Licenças dos assets 3D não documentadas (UNKNOWN). | **allowed** (download > 50 MB exige autorização). |
| **DINOv2-small** | Apache-2.0; `model.safetensors` 88 249 960 bytes; revisão `ed25f3a31f01632728cabb09d1542f84ab7b0056`. | **allowed**, mas é > 50 MB: exige autorização antes do download. Backend opcional `hf:facebook/dinov2-small@ed25f3a3…`. |

Fontes: <https://github.com/deepinsight/insightface> · <https://github.com/KwaiVGI/LivePortrait> (LICENSE) ·
<https://huggingface.co/KlingTeam/LivePortrait> · <https://github.com/ali-vilab/UniAnimate-DiT> ·
<https://aistdancedb.ongaaccel.jp/terms_of_use/> · <https://google.github.io/aistplusplus_dataset/> ·
<https://github.com/zhenzhiwang/HumanVid> · <https://huggingface.co/facebook/dinov2-small> ·
<https://github.com/facebookresearch/dinov2> (LICENSE Apache-2.0).

## Como isso é aplicado

- `scripts/datasets.py validate --sources <yaml>`: recusa fontes `blocked`, `research_only` em uso comercial,
  itens sem licença ou sem `source_url`, identidade repetida entre train/val e identidades do benchmark.
- `training/dataset.py` recusa identidades do benchmark (`forbid_identities_from` em `configs/train_adapter.yaml`).
- Nada é baixado por código do MOVA; `scripts/datasets.py plan` só descreve o que seria necessário.

## Em aberto

- Licença do projeto MOVA ainda provisória (ADR-006).
- Pexels: a licença por vídeo não basta; os Termos de Serviço proíbem download automatizado/em massa para ML.
- Outros candidatos (TikTok dataset, UBC Fashion, Champ) ainda não auditados.
