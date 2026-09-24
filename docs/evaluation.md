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

## Métricas globais — motion-v2

A normalização acima remove posição e escala de propósito. motion-v2 acrescenta o que ela esconde
(`evaluation/motion.py`), sempre em frames observados nos dois vídeos, relativos ao primeiro frame em comum:

- **trajectory**: caminho da raiz (meio dos quadris) dividido pelo torso médio de cada vídeo →
  `trajectory_error`, `final_displacement_error` (comprimentos de torso); `scale_log_error` (|log| da razão
  de tamanho aparente, detecta aproximar/afastar); `absolute_root_error` (fração da diagonal, enquadramento);
  `reference_path_length` para contexto. Personagem maior no mesmo caminho → erro 0.
- **head_rotation**: ângulo geodésico entre rotações da cabeça (matriz facial do MediaPipe projetada para
  SO(3)), absoluto e relativo ao 1º frame (desvio constante de orientação vira só erro absoluto).
- **body_orientation**: yaw do tronco pelos ombros em 3D (world landmarks), diferença circular, absoluto e relativo.
- Sem âncoras/cabeça/3D → `null` ou `UNAVAILABLE`, nunca 0.

## Revisão visual

`evaluation/review.py` gera `review/index.html`: vídeo por caso com referência | driver | gerado |
sobreposição (esqueleto do driver em ciano, gerado em magenta), tabelas por grupo de métrica, erros,
integridade, tempo/VRAM/revisão do modelo, alertas e `review.csv`. Alertas são dicas de revisão
(`REVIEW_HINTS`), não veredito.

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

## Identidade — identity-v1 (`evaluation/identity.py`)

Referência (imagem original, resolução cheia) × cada frame gerado. Sem pesos novos, sem InsightFace
(modelos não comerciais — `docs/research/licenses.md`).

- **Geometria facial:** 24 landmarks rígidos do Face Mesh (cantos dos olhos, nariz, maçãs, laterais, testa); todas
  as distâncias 3D par a par, em log, sem a escala global. Erro = média |log razão| contra a referência.
  Invariante a rotação/escala/enquadramento (testado exatamente); pontos de expressão (boca, sobrancelha, queixo)
  excluídos (testado). Detecta deriva de morfologia; **não** é reconhecimento facial.
- **Cor do rosto e do torso:** CIELAB dentro do casco dos landmarks / polígono ombros–quadris: ΔE76 da média e
  interseção de histograma a\*b\*. Torso exige quadris visíveis (senão null, como no PCK).
- **Embedding (opcional):** `--identity-embedder hf:facebook/dinov2-small@<rev>` usa um modelo **já em cache**
  (nunca baixa); sem ele o campo é UNAVAILABLE, nunca 0.
- Evidência (astronaut, MediaPipe real): mesma pessoa 0.009 · rosto 29 % mais largo 0.055 · recolorido ΔE 35 vs 1.5.
  Alertas de revisão: geometria > 0.035, ΔE rosto > 10, ΔE torso > 12 — **não calibrados em vídeo gerado**.

## Qualidade dinâmica — temporal-v1 (`evaluation/temporal.py`)

Fluxo óptico Farneback (OpenCV) entre frames consecutivos: `warp_error` (frame t deformado sobre t+1, só pixels
que passam no teste forward-backward), `static_flicker` (variação onde o fluxo < 0.5 px), `luma_flicker` (brilho
global pulsando) e `mean_flow_px`. Os mesmos números no vídeo **driver** real dão a referência: `*_ratio`.
Sintético: textura suave 0.0009 · ruído por frame 0.05 · flicker de brilho 0.157 · congelado 0 com fluxo ≈ 0
(sinalizado por `mean_flow_px_ratio` < 0.3). A aceleração de keypoints (motion-v2) não vê nada disso.

## GSB pareado cego (`evaluation/gsb.py`) — o protocolo do relatório Kling

```bash
python scripts/benchmark.py gsb --baseline <reportA.json> --candidate <reportB.json> --out outputs/gsb/A_vs_B
# avaliadores recebem index.html + ratings.csv (L/S/R por eixo); key.json fica escondido
python scripts/benchmark.py gsb-score --ratings outputs/gsb/A_vs_B/ratings.csv --key outputs/gsb/A_vs_B/key.json
```

Eixos: visual_quality, dynamic_quality, identity_preservation, motion_accuracy, expression_accuracy (os 5 do
Kling) + hands_accuracy e overall. Lado esquerdo/direito sorteado por caso (seed). Score por eixo
**GSB = (G+S)/(B+S)** do ponto de vista do candidato (> 1 favorece o candidato) + win rate G/(G+B). Preferência
humana neste benchmark, não medida objetiva. Nenhuma sessão real foi feita (não há vídeo gerado real).
