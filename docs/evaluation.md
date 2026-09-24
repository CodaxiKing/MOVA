# Evaluation

## Integridade implementada

`evaluation/video.py` rejeita arrays com quantidade/shape incorretos, NaN/Inf,
e valores fora do contrato uint8 ou float [0,1] antes da conversão.
O baseline decodifica output.mp4 completamente com OpenCV e FFmpeg estrito,
confere frames, resolução e FPS (tolerância absoluta 0.01), identifica codec e
registra duração (frames/FPS) em stats.output_validation no run.json.
Falhas impedem o registro de sucesso. O contrato é para saída de FPS constante.
PASS aqui significa integridade, não qualidade visual, identidade ou movimento.
Decoders podem ocultar certas corrupções; não se promete detectar todas.
Dependências existentes: NumPy, OpenCV, imageio-ffmpeg; CPU, sem novos pesos.
Verificado em Windows/Python 3.12; Linux não testado.

## Métricas de qualidade pendentes

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
