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

## Métricas implementadas — motion-v1

`evaluation/motion.py` compara tracks crus sem interpolar detecções ausentes.
Frames/FPS precisam coincidir; não há alinhamento temporal automático ou DTW.
Na CLI, ambos os vídeos seguem o vídeo preparado e congelado do benchmark.

- **Corpo:** xy convertido para pixels (corrige o aspecto da imagem), centrado
  no meio dos quadris e dividido pelo comprimento ombros→quadris, por frame.
  Pontos e quatro âncoras precisam de visibilidade ≥0.5. Mede articulação;
  remove translação/escala e não mede trajetória global nem proporções absolutas.
- **Mãos:** cada mão centrada no punho e escalada por punho→MCP médio em pixels;
  reporta total e esquerda/direita. Não mede posição global do punho.
- **PCK@0.1:** fração de pontos válidos do driver com erro ≤0.1 unidade normalizada.
  Ausência no output conta como incorreto. Nenhum ponto válido no driver → null.
  0.1 é um parâmetro experimental, não limiar universal de qualidade e não é
  diretamente comparável a PCK normalizado por cabeça/bounding box de outros trabalhos.
- **Erro médio:** distância apenas em pares observados; ler junto à cobertura.
  Uma queda de detecção pode diminuir artificialmente esse erro condicional.
- **Temporal:** média da norma da segunda diferença multiplicada por FPS² e
  erro dessa aceleração entre driver e output, em trios consecutivos observados
  nos dois vídeos. Sem trios → null. Não mede flicker de textura; movimento rápido
  legítimo tem aceleração alta. Congelar o personagem não é melhoria temporal.
- **Expressão:** MAE dos 52 blendshapes em frames com rosto nos dois vídeos;
  ordem dos coeficientes precisa coincidir. Não mede identidade facial.

O relatório registra hashes do benchmark, código do avaliador/extratores e pesos
MediaPipe, versões de dependências, configurações e hardware no run.json.
Casos ausentes/invalidáveis permanecem no report como failed; comparação recusa
relatórios incompletos ou condições distintas. Compare retorna deltas, sempre
REQUIRES_REVIEW; não escolhe um vencedor nem promove checkpoints.
Geração registra tempo, segundos/frame e VRAM por caso. O loader legado ainda
não fixa revisão HF: `model_revision=null`, limitação de reprodutibilidade a
resolver antes de comparar caches de pesos diferentes. Tempos herdados do
baseline têm arredondamento de 0.1 s. Não comparar desempenho entre hardwares
diferentes como se fosse ganho do modelo.

Ver `benchmark/README.md` para comandos. Sem mídia real, os testes de infraestrutura
não estabelecem qualidade do MOVA. Identidade, qualidade perceptual, fluxo óptico
e FVD continuam NOT_MEASURED/pendentes.

## Demais métricas planejadas

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
