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
ficam preservados se a execução parar. Não retoma uma execução interrompida.

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
