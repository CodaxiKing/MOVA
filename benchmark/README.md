# Benchmark MOVA

Estado: **protocolo e ferramentas implementados; conjunto real ainda não preenchido**.
`v1.draft.yaml` reserva 20 casos em dez categorias. Os caminhos são esperados,
não arquivos existentes. Os testes sintéticos não contam como esses 20 casos.

O protocolo inicial usa 256×256, 17 frames, 16 FPS e seed 42. É um teste curto
de infraestrutura (1,0625 s), insuficiente para avaliar movimentos longos.
Não há comprovação de que o baseline caiba em 8 GB. Para testar outro tamanho
ou duração, crie outra versão do protocolo; não misture resultados.

1. Forneça imagens e vídeos autorizados. Preencha fonte e licença de **cada**
   imagem/vídeo, identidade e prompt no manifesto. Os campos registram a
   declaração de uso; o código não verifica juridicamente a licença.
2. Prepare cada vídeo com o comando abaixo. Ele seleciona os primeiros frames,
   preserva o aspecto com bordas e recusa vídeos curtos ou FPS insuficiente.
   Corte o trecho de interesse antes desse comando. O arquivo preparado é a
   entrada tanto do baseline quanto da avaliação, evitando desalinhamento.
3. Execute check e freeze. Freeze confere decodificação e cria um lock com
   SHA-256 de todas as mídias e condições. Não sobrescreve locks existentes.
4. Gere na GPU, avalie e compare. Arquivos ausentes falham explicitamente.
5. Faça revisão visual usando `review-template.csv` e registre falhas em
   `failure-template.yaml`. Só depois decida qual experimento treinar.

Atalho para os passos 1–2, um caso por vez (copia a referência, prepara o vídeo, preenche fonte/licença/identidade
e mostra o que ainda falta; nunca sobrescreve mídia existente):

```bash
python scripts/benchmark.py intake --case walking-01 --reference raw/maya_front.png --motion raw/walk_clip.mp4 \
  --identity-id maya --reference-source "own photo" --reference-license "CC-BY-4.0" \
  --motion-source "own recording" --motion-license "CC-BY-4.0"
```

Depois da avaliação: `benchmark.py gsb` monta uma sessão GSB cega entre dois relatórios e `benchmark.py gsb-score`
calcula (G+S)/(B+S) por eixo (`docs/evaluation.md`).

```bash
python scripts/benchmark.py check --manifest benchmark/v1.draft.yaml
python scripts/benchmark.py prepare --video assets/raw/walking.mp4 --out assets/benchmark/v1/walking-01/motion.mp4
python scripts/benchmark.py freeze --manifest benchmark/v1.draft.yaml --out benchmark/v1.lock.json

# GPU, pesos já no cache. Sem CUDA recusa; não baixa pesos sem --allow-download.
python scripts/benchmark.py generate --lock benchmark/v1.lock.json

# Substitua RUN pelo diretório impresso por generate.
python scripts/benchmark.py evaluate --lock benchmark/v1.lock.json --outputs outputs/benchmark/RUN/videos --label baseline --contract outputs/benchmark/RUN/contract.json --generation-index outputs/benchmark/RUN/generation.json

# Substitua os caminhos pelos report.json impressos por evaluate.
python scripts/benchmark.py compare --baseline outputs/evaluation/BASELINE/report.json --candidate outputs/evaluation/CANDIDATE/report.json --out outputs/comparison.json
```

Execute duas vezes nas mesmas condições para observar a reprodutibilidade.
Cada generate executa no máximo um caso por vez, uma vez por caso; não há
repetição automática, treino ou alteração de checkpoints. Os registros parciais
ficam preservados se a execução parar. Para retomar, use o comando impresso no início
(`generate --lock ... --resume outputs/benchmark/RUN`): casos concluídos são pulados se o vídeo
não mudou; a retomada é recusada se lock, pacotes ou código de geração mudaram. `evaluate` também
aceita `--resume outputs/evaluation/RUN` (usa `report.partial.json`).

Revisão visual: `evaluate` gera `review/index.html` ao lado do `report.json` (vídeo
referência | driver | gerado | sobreposição de esqueletos, métricas por grupo, alertas e
`review.csv` pré-preenchido). Para refazer: `python scripts/benchmark.py review --report <report.json>`.
Os alertas (`evaluation/review.py::REVIEW_HINTS`) só indicam onde olhar primeiro.

Para avaliar vídeos produzidos externamente, organize `<id>.mp4` e forneça um
JSON de condições com `--contract`. Sem generation-index, tempo e VRAM ficam
null e a origem é `user_declared`: as condições não são verificadas pelo código.
Compare apenas condições realmente equivalentes. Um lock identifica os dados,
não é uma assinatura de segurança contra alterações deliberadas.

Reserve estas identidades para teste: não use os casos para treinamento nem
para selecionar repetidamente hiperparâmetros. Monte um conjunto de validação
separado para desenvolvimento. Treino/dataset loader ainda não implementados.

Revisão cega: uma pessoa organiza as saídas como A/B e guarda a correspondência
separada do avaliador; alterne os lados. O CSV não automatiza esse cegamento.
Examine todos os frames. Use PASS/PARTIAL/FAIL por eixo com justificativa;
não transforme null em zero nem esconda casos difíceis.
