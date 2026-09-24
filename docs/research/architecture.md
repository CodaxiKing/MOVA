# Architecture Research — Hipótese Modular

> Esta é uma **hipótese de engenharia** do MOVA. Não afirma ser a arquitetura interna de nenhum sistema proprietário.
> Arquitetura efetivamente implementada: `docs/architecture.md`.

## Hipótese

```text
Reference image(s) ──► Identity Encoder ──► identity tokens ─────────────┐
                                                                          │
Motion video ─┬─► Body extractor ──► Body Motion Encoder ──┐              │
              ├─► Face extractor ──► Face Motion Encoder ──┼─► Motion ────┤
              └─► Hand extractor ──► Hand Motion Encoder ──┘   Fusion     │
                                                                  │       │
                                                       motion residuals   │
                                                                  ▼       ▼
                                     Wan2.1 DiT 1.3B (CONGELADO) + Motion Adapter + Identity Injection
                                                                  │
                                                             Wan-VAE decode
                                                                  ▼
                                                           Generated video
```

## Evidência pública para cada bloco

| Bloco | Evidência (FACT) | Nossa escolha inicial (HYPOTHESIS) |
|---|---|---|
| Body motion como sinal espacial alinhado | Wan-Animate: "spatially-aligned skeleton signals"; Animate Anyone: Pose Guider; VACE: control video | Renderizar esqueleto → encoder conv 3D leve → mesmo grid dos latentes (T/4, H/8, W/8) → patchify → residual |
| Face como features implícitas compactas | Wan-Animate: "implicit facial features extracted from source images"; LivePortrait keypoints implícitos | Blendshapes (52) + pose da cabeça (6D) + landmarks normalizados → MLP → tokens por frame |
| Mãos separadas | MimicMotion: amplificação de loss em mãos; Kling: representações heterogêneas por região | Landmarks 21×2 (2D+world 3D) → MLP → tokens; + loss ponderada por região |
| Fusão | Kling: "progressive multi-stage training"; UniAnimate-DiT: pose encoder + LoRA | Corpo como residual espacial; face/mãos como tokens via cross-attention leve nos blocos |
| Identidade | VACE: reference images; Wan-Animate: referência na sequência; Kling: subject library multi-view | 1ª fase: usar mecanismo nativo do VACE (reference latents). Depois: Identity Encoder próprio (CLIP/DINOv2 + face embedding) |
| Backbone congelado + adapters | UniAnimate-DiT (LoRA no Wan), VACE (blocos de contexto) | Adapter zero-init por bloco; LoRA opcional rank 16–64 |

## Perguntas de pesquisa abertas

1. Pose renderizada (imagem) vs keypoints vetoriais (tokens) para o corpo: qual dá melhor precisão por MB de VRAM?
2. 2D vs 3D (MediaPipe world) melhora casos de rotação do corpo? (Kling cita 3D awareness — I2 em `kling_analysis.md`)
3. Retargeting de proporções (comprimento de ossos) antes de condicionar reduz deformação de identidade?
4. Face: blendshapes são suficientes para expressão ou precisamos de features implícitas (LivePortrait)?
5. Quanto do resultado vem do adapter vs do mecanismo VACE já existente?

Cada pergunta vira um experimento em `docs/experiments/`.
