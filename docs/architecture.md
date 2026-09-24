# Architecture

Hipótese e evidências: `docs/research/architecture.md`. Decisões: `DECISIONS.md`.

## Implementado (2026-09-24)

```text
                         ┌───────────────────────── preprocessing/pipeline.py ─────────────────────────┐
motion.mp4 ─► decode ─►  │ BodyExtractor (MediaPipe Pose)  → body_motion.pt  + body_preview.mp4       │
                         │ FaceExtractor (Face Landmarker) → face_motion.pt  + face_preview.mp4       │
                         │ HandExtractor (Hand Landmarker) → hand_motion.pt  + hands_preview.mp4      │
                         │   (lado da mão = punho do corpo mais próximo)                               │
                         │ render OpenPose-18 + mãos + pontos de rosto → pose_openpose.mp4            │
                         └─────────────────────────────────────────────────────────────────────────────┘

reference.png ─► letterbox(W×H) ─┐
pose_openpose.mp4 ─► 4k+1 frames ├─► WanVACEPipeline (Wan2.1-VACE-1.3B, Diffusers)
prompt ─► UMT5 (CPU, 1×) ─► cache┘      offload + VAE tiling  ─► output.mp4 / side_by_side.mp4
```

## Planejado

```text
Reference image(s) ─► Identity Encoder ─► identity tokens ───────────────────────┐
body_motion.pt  ─► Body Motion Encoder  (conv 3D sobre pose renderizada ou MLP)  │
face_motion.pt  ─► Face Motion Encoder  (blendshapes + head rot6d → tokens)      │
hand_motion.pt  ─► Hand Motion Encoder  (21×2×3 → tokens)                        │
                     └──► Projection ─► Temporal Attention ─► Motion Fusion       │
                                                  │                               │
                                   Motion Adapter (residual zero-init por bloco)  │
                                                  ▼                               ▼
                               Wan2.1 DiT 1.3B CONGELADO  ◄── Identity Injection
                                                  ▼
                                            Wan-VAE decode
```

Treináveis inicialmente: Motion Encoder, Projection, Motion Adapter. Backbone congelado.

### Contratos que os módulos futuros devem respeitar
- Latentes Wan: `(B, 16, (T-1)/4+1, H/8, W/8)`; tokens após patch 1×2×2; dim oculta 1536 (1.3B), 30 blocos.
- `num_frames = 4k+1`; H, W múltiplos de 16.
- Adapter com saída inicializada em zero → no passo 0 o modelo é idêntico ao backbone.
- Entradas de movimento vêm dos `.pt` descritos em `docs/pipeline.md` (FORMAT_VERSION=1).
