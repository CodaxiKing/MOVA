# Research Sources

Registro de todas as fontes consultadas sobre modelos externos.
Convenção de confiabilidade usada em todo o projeto:

| Tag | Significado |
|---|---|
| **FACT** | Confirmado em fonte primária (paper, GitHub/HF oficial, documentação oficial) |
| **INFERENCE** | Dedução nossa a partir de fatos públicos |
| **HYPOTHESIS** | Ideia de engenharia ainda não testada |
| **UNKNOWN** | Não pôde ser confirmado publicamente |

Última verificação geral: **2026-09-24** (Claude Code, via WebFetch/WebSearch das páginas oficiais).
Números de VRAM são os publicados pelos autores; **nenhum foi medido por nós ainda** (a máquina desta sessão não tem GPU NVIDIA).

---

## Wan2.1 (Wan-AI / Alibaba Tongyi)

- Repository: https://github.com/Wan-Video/Wan2.1
- Paper: https://arxiv.org/abs/2503.20314 ("Wan: Open and Advanced Large-Scale Video Generative Models")
- Official Documentation: README do repo; Diffusers: https://huggingface.co/docs/diffusers/main/en/api/pipelines/wan
- License: **Apache 2.0** (FACT)
- Weights: HF `Wan-AI/Wan2.1-T2V-1.3B`, `Wan-AI/Wan2.1-VACE-1.3B`, versões `-diffusers`; 14B T2V/I2V/VACE/FLF2V (FACT)
- Training Code: não no repo oficial; treino via DiffSynth-Studio e VideoX-Fun (FACT)
- Inference Code: sim (FACT)
- Architecture: DiT com flow matching; Wan-VAE 3D causal; texto via UMT5 + cross-attention; MLP de timestep compartilhado. 1.3B: dim 1536, 12 heads, 30 layers, FFN 8960. 14B: dim 5120, 40 heads, 40 layers (FACT)
- VRAM: T2V-1.3B 480P = **8.19 GB** em RTX 4090, ~4 min para 5 s de vídeo, com `--offload_model True --t5_cpu` (FACT)
- Resolution: 480P (1.3B); 720P no 14B (FACT)
- Frames: 81 frames @16 fps por padrão; num_frames = 4k+1 (FACT, restrição do VAE temporal 4x)
- Motion Control: não nativo no T2V; via VACE / Fun-Control (FACT)
- Identity Preservation: via VACE reference images / I2V (FACT)
- Face Control / Hand Control: não dedicado (FACT)
- Fine-tuning: sim (DiffSynth, VideoX-Fun) (FACT)
- LoRA: sim (FACT)
- Local Execution: sim; Windows não documentado oficialmente, mas funciona via Diffusers/ComfyUI (INFERENCE)
- Advantages: backbone DiT pequeno (1.3B) treinável em consumer GPU; Apache 2.0; ecossistema enorme
- Limitations: UMT5-XXL (~5.7B params) é pesado; qualidade do 1.3B inferior ao 14B
- Relevance to MOVA: **backbone escolhido** (ver `backbone_selection.md`)

## Wan2.1-VACE-1.3B

- Repository: https://github.com/ali-vilab/VACE ; HF https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B e `Wan-AI/Wan2.1-VACE-1.3B-diffusers`
- Paper: https://arxiv.org/abs/2503.07598 (VACE: All-in-One Video Creation and Editing)
- License: **Apache 2.0** (FACT)
- Architecture: Wan2.1-1.3B + "context blocks" (VACE layers) que injetam latentes de controle (vídeo de controle + máscara + imagens de referência) de forma aditiva em camadas selecionadas; `conditioning_scale` por camada (FACT, Diffusers API)
- Tasks: control-to-video (pose, depth, flow...), reference-to-video, V2V, inpainting/outpainting, composição "animate anything" (FACT)
- VRAM: não publicado separadamente; base 1.3B = 8.19 GB (FACT). Diffusers oferece group offloading + VAE tiling (FACT)
- Resolution/Frames: 480P estável, 720P menos estável; 81 frames (FACT)
- Fine-tuning/LoRA: via DiffSynth-Studio (FACT)
- Relevance to MOVA: **baseline da Fase 1** (pose video + reference image → vídeo)

## Wan2.1-Fun-1.3B-Control / Wan2.1-Fun-V1.1-1.3B-Control (alibaba-pai)

- Repository: https://github.com/aigc-apps/VideoX-Fun ; HF https://huggingface.co/alibaba-pai/Wan2.1-Fun-V1.1-1.3B-Control
- License: **Apache 2.0** (FACT)
- Controls: Canny, Depth, Pose, MLSD, trajetória; V1.1 suporta "reference image + control" (FACT)
- Resolution: multi-resolução 512/768/1024; 81 frames @16 fps (FACT)
- Weights: **19.0 GB** total no repo HF (FACT, inclui encoder de texto)
- Low-VRAM: `model_cpu_offload`, `model_cpu_offload_and_qfloat8`, `sequential_cpu_offload` (FACT)
- Training Code: full e LoRA (FACT)
- Windows: testado em Windows 10, Python 3.10–3.11, CUDA 11.8–12.1; README cita "3060 12GB" (FACT)
- Relevance to MOVA: **baseline alternativo B** e referência de como treinar controle em Wan-1.3B

## Wan2.2-Animate-14B (Wan-Animate)

- Repository: https://github.com/Wan-Video/Wan2.2 ; HF https://huggingface.co/Wan-AI/Wan2.2-Animate-14B (+ `-Diffusers`)
- Paper: https://arxiv.org/abs/2509.14055 ; Project: https://humanaigc.github.io/wan-animate
- License: **Apache 2.0** (FACT)
- Architecture (FACT, abstract): construído sobre Wan; "modified input paradigm" que diferencia condições de referência e regiões a gerar; **sinais de esqueleto espacialmente alinhados** para corpo; **features faciais implícitas extraídas de imagens do rosto** para expressão; **Relighting LoRA** para o modo replacement.
- Diffusers `WanAnimatePipeline`: entradas `image`, `pose_video`, `face_video`, (`background_video`, `mask_video` no replacement); `segment_frame_length=77`, `prev_segment_conditioning_frames=1` (FACT)
- Modes: animation / replacement (FACT)
- VRAM: single-GPU requer `offload_model True`; sem números oficiais para consumer (FACT). Comunidade roda via GGUF + block swap (ComfyUI-WanVideoWrapper) (INFERENCE, fontes secundárias)
- Resolution: 480P/720P, 24 fps (FACT)
- Training Code: **não liberado** (FACT)
- Weights: ~17B params BF16 (FACT, card HF)
- Relevance to MOVA: **arquitetura de referência** (separação corpo/rosto/identidade); inviável como backbone treinável em 8 GB

## Wan-Animate-2 (Wan2.2-Animate-2-14B)

- Repository: https://github.com/Wan-Video/Wan-Animate-2 ; HF https://huggingface.co/Wan-AI/Wan2.2-Animate-2-14B
- Paper: https://arxiv.org/abs/2608.06009 ; Project: https://humanaigc.github.io/wan-animate-2/
- Release: 2026-08-07 (FACT, notícia + repo)
- License: **Apache 2.0** (FACT)
- Architecture (FACT, abstract): framework end-to-end que **consome o vídeo de condução diretamente** num DiT redesenhado, **sem extratores intermediários de movimento**; controle de viewpoint por texto; variante **Lite** para streaming em tempo real via teacher forcing + Self-Forcing distillation.
- Variants: Base 14B (40 steps), Distillation (10 steps), versões quantizadas (FACT)
- VRAM: default 8× A800 para 720P; 480P testado em 2× A800 (FACT). Nenhum número consumer.
- Training Code / LoRA: não liberados (FACT)
- OS: Linux, Python 3.11, PyTorch 2.7 + CUDA 12.6, flash-attn (FACT)
- Relevance to MOVA: sinal de tendência — extratores explícitos podem ser substituídos por encoder aprendido do vídeo. Útil para a Fase de "latent motion features". Inviável em 8 GB. (INFERENCE)

## MimicMotion (Tencent)

- Repository: https://github.com/Tencent/MimicMotion
- Paper: https://arxiv.org/abs/2406.19680
- License: código **Apache 2.0** (FACT). Backbone SVD-XT-1.1 sob **Stability AI Community License** (gated; grátis < US$1M receita/ano; atribuição) (FACT, card HF)
- Architecture: SVD (U-Net) + pose guidance com confiança (DWPose), regional loss amplification (mãos), progressive latent fusion para vídeos longos (FACT, paper)
- VRAM: v1.1 72 frames = 16 GB (4060 Ti); v1 16 frames: U-Net 8 GB, VAE decoder 16 GB (pode ir p/ CPU) (FACT)
- Resolution: até 576×1024, 72 frames (FACT)
- Training Code: **não** (FACT)
- Relevance to MOVA: baseline U-Net para comparação (opcional); não serve como backbone treinável

## Animate Anyone (Alibaba HumanAIGC) / Moore-AnimateAnyone

- Paper: https://arxiv.org/abs/2311.17117 (código oficial nunca liberado — FACT)
- Reprodução: https://github.com/MooreThreads/Moore-AnimateAnyone — **Apache 2.0** (FACT)
- Architecture: SD1.5 + ReferenceNet (identidade via spatial attention) + Pose Guider (conv leve somado ao ruído) + motion module (AnimateDiff) (FACT)
- Training Code: sim, 2 estágios (FACT)
- VRAM: ≥16 GB para o app Gradio (FACT)
- Resolution: 512×768 (FACT)
- Windows: fork comunitário https://github.com/sdbds/Moore-AnimateAnyone-for-windows (FACT)
- Relevance to MOVA: referência didática de ReferenceNet + Pose Guider; U-Net antigo

## MagicAnimate (ByteDance / NUS)

- Repository: https://github.com/magic-research/magic-animate — **BSD-3-Clause** (FACT)
- Architecture: SD1.5 + appearance encoder + DensePose ControlNet + temporal attention (FACT)
- Training Code: **não** (FACT); sem atualização desde 2023-12 (FACT)
- Relevance to MOVA: baixa (histórico)

## UniAnimate-DiT (Alibaba DAMO)

- Repository: https://github.com/ali-vilab/UniAnimate-DiT
- License: **UNKNOWN** (não exibida no README consultado)
- Architecture: LoRA sobre Wan2.1-14B-I2V-720P + pose encoder (DWPose) (FACT)
- VRAM: 480P ~23 GB, 14 GB com `num_persistent_param_in_dit=0` (FACT)
- Training Code: LoRA (DiffSynth) (FACT); recomenda ~1000 vídeos para finetune (FACT)
- Relevance to MOVA: prova de que "pose encoder + LoRA num Wan congelado" funciona; replicaremos a ideia no 1.3B (INFERENCE)

## StableAnimator

- Repository: https://github.com/Francis-Rings/StableAnimator — **MIT** (código) (FACT); backbone SVD (Stability Community License)
- Identity: Face Encoder global content-aware + ArcFace (antelopev2 / InsightFace — licença não comercial dos pesos, INFERENCE a verificar) + otimização baseada em HJB no denoising (FACT)
- VRAM: inferência 16 frames 512×512 = 8 GB; treino 40–70 GB (FACT)
- Relevance to MOVA: design de identity/face encoder

## SteadyDancer (MCG-NJU)

- Repository: https://github.com/MCG-NJU/SteadyDancer — **Apache 2.0** (FACT)
- Architecture: I2V com preservação do primeiro frame sobre Wan2.1-I2V (14B) + modelagem de pose (FACT)
- Relevance to MOVA: paradigma I2V (primeiro frame = identidade) como alternativa à referência separada

## LivePortrait (Kuaishou / KwaiVGI)

- Repository: https://github.com/KwaiVGI/LivePortrait — código aberto (ver LICENSE); componentes InsightFace com restrições próprias (FACT)
- Architecture: keypoints implícitos, motion extractor, appearance extractor, stitching/retargeting (FACT)
- Windows: suportado (FACT); Training Code: não (FACT)
- Relevance to MOVA: **candidato ao Face Motion Encoder** (keypoints implícitos + expressão) — licença dos pesos precisa ser auditada antes de uso (TODO)

## Kling-MotionControl (Kuaishou) — proprietário

- Technical report: https://arxiv.org/abs/2603.03160 (HTML: https://arxiv.org/html/2603.03160v1)
- API pública (terceiros): https://replicate.com/kwaivgi/kling-v3-motion-control
- Code/Weights: **não disponíveis** (FACT). Análise completa em `kling_analysis.md`.

## Motion Mirror

- Busca por "Motion Mirror" não encontrou projeto de animação de personagem por difusão; o resultado relevante é um projeto maker (Hackster.io, rastreamento facial em microcontrolador). **UNKNOWN** se o usuário se refere a outro projeto — pedir link.
- https://www.hackster.io/ishamsu/motion-mirror-530eb8

## Ferramentas de infraestrutura

| Ferramenta | Link | Licença | Uso no MOVA |
|---|---|---|---|
| Diffusers (`WanVACEPipeline`, `WanAnimatePipeline`) | https://huggingface.co/docs/diffusers/main/en/api/pipelines/wan | Apache 2.0 | inferência baseline, offload, VAE tiling |
| DiffSynth-Studio | https://github.com/modelscope/DiffSynth-Studio | Apache 2.0 | treino LoRA Wan; afirma treino do 1.3B com ~6 GB via offload (FACT, README; não verificado por nós) |
| VideoX-Fun | https://github.com/aigc-apps/VideoX-Fun | Apache 2.0 | referência de treino de controle |
| DWPose | https://github.com/IDEA-Research/DWPose | Apache 2.0 | 133 keypoints whole-body (corpo, pés, 68 face, 21×2 mãos); ONNX; formato esperado por VACE/UniAnimate |
| MediaPipe Tasks (Pose/Face/Hand Landmarker) | https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker | Apache 2.0 | extração CPU: 33 pose + world 3D; 478 face + 52 blendshapes + matriz de pose da cabeça; 21 landmarks/mão + world 3D |
| ComfyUI-WanVideoWrapper / GGUF | https://huggingface.co/city96/Wan2.1-T2V-14B-gguf | vários | rota para rodar 14B em 8 GB (block swap) — fonte secundária |

## Datasets

Ver `docs/dataset.md` para a tabela completa. Fontes:
- HumanVid: https://github.com/zhenzhiwang/HumanVid — CC-BY-4.0 (código + sintéticos); vídeos reais seguem termos do Pexels (FACT)
- TikTok Dataset (Jafarian & Park, CVPR 2021): https://www.kaggle.com/datasets/yasaminjafarian/tiktokdataset — 340 clipes de 10–15 s, 30 fps, >100K frames (FACT); licença **UNKNOWN** (conteúdo de terceiros do TikTok → tratar como pesquisa apenas)
- AIST++: https://google.github.io/aistplusplus_dataset/ — anotações CC BY 4.0; vídeos do AIST Dance DB com termos próprios (INFERENCE, página de factsheet retornou 404 — reverificar)
- OpenHumanVid: https://arxiv.org/abs/2412.00115

## Avaliação — consultado em 2026-09-24

- FACT: PCK depende do limiar e fator de normalização; implementação primária
  consultada: https://github.com/open-mmlab/mmpose/blob/main/mmpose/evaluation/metrics/keypoint_2d_metrics.py
- FACT: MediaPipe expõe 52 coeficientes de blendshape:
  https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/drawing_styles/face_landmarker/Blendshapes
- HYPOTHESIS de engenharia: PCK com torso/palma e limiar 0.1 fornece diagnóstico
  inicial útil no MOVA. Não validado como limiar perceptual de sucesso.

## Fontes secundárias históricas

- https://www.opensourceforu.com/2026/08/alibaba-open-sources-wan-animate-2/
- https://comfyui-wiki.com/en/news/2026-08-07-wan-animate-2
- https://github.com/Cordux/ComfyUI-Wan2.2-workflow
- https://www.financialcontent.com/article/abnewswire-2026-9-12-kling-ai-launches-kling-30-... (press release, marketing)
