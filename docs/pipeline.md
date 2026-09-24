# Motion Extraction Pipeline (Phase 2)

```bash
.venv/Scripts/python scripts/extract_motion.py --video assets/motion/dance.mp4 --out outputs/motion/dance \
    [--set target_fps=16] [--set max_frames=64] [--set pose_model=pose_landmarker_heavy]
```

Config: `configs/extraction.yaml`. Modelos MediaPipe (`checkpoints/mediapipe/*.task`, ~21 MB total) são baixados automaticamente na primeira execução.

## Saídas

| Arquivo | Conteúdo |
|---|---|
| `body_motion.pt` | corpo |
| `face_motion.pt` | rosto |
| `hand_motion.pt` | mãos |
| `body_preview.mp4`, `face_preview.mp4`, `hands_preview.mp4` | overlays no vídeo original |
| `pose_openpose.mp4` | controle estilo OpenPose (fundo preto) — entrada do baseline VACE |
| `summary.json` | taxa de detecção, jitter, tempos |

Todos os `.pt` são dicts (`torch.load(p, weights_only=False)`) com `meta = {format_version, source, fps, width, height, num_frames}`.

### body_motion.pt (FORMAT_VERSION 1)
| Chave | Shape | Descrição |
|---|---|---|
| `kp2d` | (T, 33, 4) | x, y normalizados [0,1] na imagem, z relativo, visibilidade |
| `kp3d_world` | (T, 33, 3) | metros, centrado no quadril |
| `present` | (T,) bool | pose detectada |
| `features.xy_filled` | (T, 33, 2) | xy com frames faltantes interpolados |
| `features.xy_normalized` | (T, 33, 2) | centrado no meio do quadril, escala = comprimento do torso |
| `features.center`, `features.scale` | (T, 2), (T,) | para desfazer a normalização |
| `features.velocity_normalized`, `acceleration_normalized` | (T, 33, 2) | por segundo |
| `features.world_filled`, `velocity_world` | (T, 33, 3) | 3D |

### face_motion.pt
| Chave | Shape | Descrição |
|---|---|---|
| `landmarks` | (T, 478, 3) | mesh normalizado |
| `blendshapes` | (T, 52) | coeficientes ARKit-style; nomes em `blendshape_names` |
| `head_transform` | (T, 4, 4) | matriz de transformação facial |
| `head_euler_deg` | (T, 3) | pitch, yaw, roll |
| `head_translation` | (T, 3) | |
| `features.head_rot6d` | (T, 6) | rotação contínua 6D |
| `features.blendshapes_filled`, `blendshapes_velocity` | (T, 52) | |

### hand_motion.pt
| Chave | Shape | Descrição |
|---|---|---|
| `kp2d` | (T, 2, 21, 3) | índice 0 = mão esquerda **da pessoa**, 1 = direita |
| `kp3d_world` | (T, 2, 21, 3) | metros |
| `present` | (T, 2) | |
| `handedness_score` | (T, 2) | |
| `features.{left,right}_xy_filled` | (T, 21, 2) | |
| `features.{left,right}_world_normalized` | (T, 21, 3) | centrado no punho, escala punho→MCP médio |
| `features.{left,right}_velocity_world` | (T, 21, 3) | |

## Atribuição esquerda/direita das mãos
MediaPipe assume imagem espelhada (selfie). Com corpo detectado, cada mão é associada ao punho do corpo mais próximo; sem corpo, usa o rótulo de handedness invertido (`mirrored_input=false`).

## Desempenho medido (CPU i7-1165G7, 512×512)
Vídeo sintético "astronaut", 24 frames: corpo 0.48 s, rosto 0.27 s, mãos 0.33 s (≈ 45 ms/frame total).
