# Datasets

Nada foi baixado. Não baixar datasets grandes sem autorização.

Benchmark separado do treino: `benchmark/v1.draft.yaml` contém 20 vagas com
fonte/licença independentes para referência e movimento e identity_id obrigatório.
Os dados ainda precisam ser fornecidos. `benchmark/README.md` descreve preparação,
congelamento e revisão. Não usar suas identidades no treino; criar validação
separada para seleção de hiperparâmetros. A verificação automatizada de vazamento
entre splits depende do futuro manifesto de treinamento.

| Dataset | Origem | Licença | Tamanho | Resolução | Conteúdo | Treino legal? |
|---|---|---|---|---|---|---|
| **HumanVid** | https://github.com/zhenzhiwang/HumanVid | CC-BY-4.0 (código + sintéticos UE); vídeos reais seguem termos do Pexels | real (Pexels) + sintético; tamanho exato não confirmado | variada (vertical/horizontal) | humanos em movimento com parâmetros de câmera | **Sim** só para os sintéticos (Hub). **Não** para os reais: os Termos do Pexels proíbem download automatizado para ML (reauditado 2026-09-24) |
| **TikTok Dataset** (Jafarian & Park, CVPR 2021) | https://www.kaggle.com/datasets/yasaminjafarian/tiktokdataset | **UNKNOWN** (conteúdo de terceiros) | 340 clipes, 10–15 s, 30 fps, >100K frames | ~1080×604 (INFERENCE) | dança, 1 pessoa | Só pesquisa/avaliação; **não** para modelo distribuído |
| **AIST++** | https://google.github.io/aistplusplus_dataset/ | anotações CC BY 4.0; vídeos AIST Dance DB (termos próprios) — reverificar | ~1 400 sequências, 10 gêneros de dança, multi-view (INFERENCE, a confirmar) | 1920×1080 | dança com 3D keypoints/SMPL | Anotações sim; vídeos: verificar termos |
| **OpenHumanVid** | https://arxiv.org/abs/2412.00115 | a verificar | grande | alta | vídeos humanos com legendas | a verificar |
| **UBC Fashion** | paper DwNet | pesquisa | ~500 vídeos | 720×940 | moda, rotação lenta | pesquisa apenas (INFERENCE) |
| **Vídeos próprios** | gravados pelo usuário | do usuário | — | — | qualquer | **Sim** |
| **Pexels / Pixabay** (curadoria manual) | sites | licença livre dos sites | — | — | dança, esporte | Sim, respeitando termos |

## Dataset experimental (Fase 3)

- 20–100 clipes curtos de 1 pessoa, corpo inteiro, câmera fixa. Fonte preferida: vídeos próprios + Pexels + HumanVid sintético (licenças limpas).
- Processamento: 256×256, 17 frames, fps 16, `.pt` de movimento pré-extraídos, latentes VAE pré-computados.
- Split: 80/20 por identidade (nenhuma pessoa do teste no treino).
- Layout:
```text
datasets/
├── raw/<clip_id>.mp4                (git-ignored)
├── processed/<clip_id>/{body,face,hand}_motion.pt, latents.pt, ref.png   (git-ignored)
└── manifests/{train,val}.jsonl      (versionado: id, fonte, licença, duração)
```
