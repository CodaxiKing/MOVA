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

### body_motion.pt (FORMAT_VERSION 2; ver "Formato v2" abaixo)
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

## Formato v2 (2026-09-24)

`FORMAT_VERSION = 2`. Mudança de semântica só em `hand_motion.pt`, com `hand_postprocess: true` (padrão):

| chave | v1 | v2 |
|---|---|---|
| `kp2d`, `kp3d_world`, `present`, `handedness_score` | detecção crua | após rejeição de mão longe do punho e correção de troca E/D |
| `kp2d_raw`, `present_raw` | — | detecção crua (preservada) |
| `kp2d_filled`, `filled` | — | lacunas internas ≤ `hand_max_gap` (3) interpoladas; **só para o vídeo de controle** |
| `features.*_smoothed` (corpo, mãos, `landmarks_smoothed` do rosto) | — | só com `smoothing: one_euro` |

`present` continua significando "detectado": frames preenchidos nunca contam como evidência na avaliação.
A avaliação aceita v1 e v2, mas exige a mesma versão nos dois lados.

## Vídeo de controle a partir das tracks

`pose_openpose.mp4` é desenhado depois da extração (`preprocessing/control.py`), não mais ao vivo. Com os
padrões e `hand_postprocess: false` é pixel a pixel idêntico ao desenho antigo (verificado; teste em
`tests/test_signal.py`). Isso permite suavizar, limpar as mãos e **retargetar** antes de desenhar.

## Limpeza do sinal

- `smoothing: one_euro` (`preprocessing/filters.py`, Casiez 2012), estado reiniciado após cada lacuna. Padrão
  `none`: na calibração sintética (EXP-006) o preset 2.0/100 reduz ~25 % o jitter em movimento calmo, mas aumenta
  ~50 % o erro em movimento rápido (atraso). Decidir com vídeo real (EXP-002).
- Mãos (`preprocessing/hands_post.py`): mão cujo punho está a mais de 1 antebraço do punho do corpo é descartada;
  troca E/D corrigida por continuidade quando os punhos do corpo não são confiáveis; lacunas curtas preenchidas só
  no controle. `summary.json` → `hands_postprocess` conta cada alteração.
- `scripts/calibrate_extraction.py --video X`: varre `min_confidence` e mostra detecção × jitter por stream.

## Retargeting de proporções (`preprocessing/retarget.py`)

`child' = parent' + (child − parent) · L_ref / L_driver` ao longo da árvore a partir da pelve. Comprimentos-alvo do
esqueleto **3D métrico** da referência (MediaPipe world) → comprimentos 3D de saída = os da personagem; direções,
timing e escorço 2D = os do ator; caminho da raiz escala com o torso; juntas ausentes propagam para os filhos;
mãos seguem o punho, rosto segue o nariz. Saída desenhada no enquadramento da referência.
Uso: `inputs.retarget: true` em `mova infer`, ou `mova preprocess --video X --out Y --retarget-to ref.png`
(escreve `Y/retargeted/` com `.pt`, `pose_openpose.mp4` e `retarget.json`). Exige torso visível na referência e
no driver; senão erro claro (o retrato astronaut, sem quadris, é recusado — testado).
