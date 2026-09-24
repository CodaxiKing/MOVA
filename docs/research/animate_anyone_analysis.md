# Animate Anyone / Moore-AnimateAnyone / MagicAnimate — Análise

Última verificação: 2026-09-24.

## Animate Anyone (paper 2311.17117, Alibaba HumanAIGC)
**FACT:** ReferenceNet (cópia do U-Net SD que extrai features da imagem de referência, fundidas via spatial attention), Pose Guider (conv leve cujo output é somado ao latente ruidoso), módulo temporal. **Código oficial nunca liberado.**

## Moore-AnimateAnyone
**FACT:** reprodução Apache 2.0; SD1.5 + ReferenceNet + Pose Guider + motion module (AnimateDiff); **código de treino em 2 estágios** (1: imagem — pose guider + reference net; 2: motion module); DWPose; 512×768; ≥16 GB para Gradio; fork Windows comunitário. Os autores relatam ~80% da qualidade do paper.

## MagicAnimate
**FACT:** BSD-3; SD1.5 + appearance encoder + DensePose ControlNet + temporal attention; sem código de treino; sem atualizações desde 2023-12.

## INFERENCE — o que levamos
- **Treino em 2 estágios** (espacial → temporal) é um padrão comprovado e barato: primeiro aprender pose→frame, depois consistência temporal. Adaptamos para o DiT: estágio A em poucos frames, estágio B em mais frames.
- **Pose Guider aditivo e leve** (conv somado ao latente) é o design mais barato de controle; é nosso ponto de partida para o Body Motion Encoder.
- **ReferenceNet** duplica o backbone → caro demais em 8 GB. Preferir identidade via tokens (cross-attention/concat) ou referência como frame extra (paradigma VACE/Wan-Animate).
- Arquiteturas U-Net SD1.5 são obsoletas frente aos DiTs de vídeo; não são candidatas a backbone.
