# MimicMotion — Análise Técnica

Última verificação: 2026-09-24. Repo: https://github.com/Tencent/MimicMotion · Paper: https://arxiv.org/abs/2406.19680

## FACT
- Backbone: Stable Video Diffusion `img2vid-xt-1-1` (U-Net espaço-temporal, ~1.5B).
- Condição de pose: DWPose (yolox_l + dw-ll_ucoco_384), **confidence-aware pose guidance** (keypoints com baixa confiança têm menos peso).
- **Regional loss amplification** em regiões de mão (ideia de loss que queremos testar — ver `docs/training.md`).
- **Progressive latent fusion** para vídeos longos (sobreposição de janelas).
- v1.1: 72 frames, 576×1024, 16 GB (4060 Ti). v1: 16 frames, U-Net 8 GB, VAE decoder 16 GB (pode ser movido para CPU).
- Código Apache 2.0; SVD sob Stability AI Community License (gated, limite US$1M, atribuição obrigatória).
- **Sem código de treino.**

## INFERENCE
- Rodaria na RTX 3060 8 GB em modo v1 (16 frames) com VAE em CPU ou decode em tiles.
- Útil como **segundo baseline** (U-Net) para comparação; não serve como base para nossos adapters (sem treino, licença SVD mais restritiva, arquitetura U-Net oposta ao objetivo DiT).

## Ideias reaproveitáveis
1. Ponderação por confiança do keypoint no sinal de controle.
2. Amplificação de loss nas regiões de mão/rosto.
3. Fusão progressiva de latentes em janelas sobrepostas para vídeos longos.
