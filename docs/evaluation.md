# Evaluation (planejado — `evaluation/` vazio)

Cada eixo é avaliado separadamente (princípio também usado na avaliação do Kling-MotionControl tech report: identidade, precisão de movimento, expressão, qualidade dinâmica e visual).

| Eixo | Métrica | Como | Dependências |
|---|---|---|---|
| Motion / Pose Accuracy | erro médio de keypoints normalizados e PCK@0.1 entre pose re-extraída do vídeo gerado e pose de condução | nosso `preprocessing` (MediaPipe) aplicado ao output | já existe |
| Hand Accuracy | idem, só 21×2 pontos das mãos + taxa de detecção de mãos no output | idem | já existe |
| Face / Expression | distância de blendshapes e erro de head pose (graus) output vs condução | Face Landmarker | já existe |
| Identity Consistency | similaridade de embedding facial referência × frames; CLIP/DINOv2 image similarity para roupa/corpo | modelo de embedding | escolher modelo com licença adequada |
| Temporal Consistency | jitter de keypoints (`features.temporal_jitter`); erro de warp com fluxo óptico entre frames | OpenCV Farneback (sem pesos) | já existe |
| Video Quality | FVD (se houver conjunto de referência), CLIP-score do prompt, avaliação visual lado a lado | — | a definir |

Toda avaliação grava `experiments/runs/<id>/run.json` e compara BASELINE × MOVA no mesmo conjunto de pares (referência, movimento).
