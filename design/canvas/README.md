# Canvas de design — MOVA Motion Control

Protótipo de interface (site + estúdio) para testar a experiência de Motion Control.
**É só design/simulação: nenhum modelo roda no canvas.**

- **Abrir o canvas:** dê dois cliques em `Abrir canvas MOVA.url`
  (ou acesse https://claude.ai/artifact/L7g64Yo2eYKy2jXcAeL6gW — privado, exige login na sua conta Claude).
- **Fontes das telas** (cópia de 2026-09-24, versão 5 do canvas), em `project/`:
  - `Home.dc.html` — landing page (Início)
  - `Main.dc.html` — Estúdio (upload, parâmetros, geração simulada, comando `mova infer`)
  - `Motion.dc.html` — extração de movimento (corpo/rosto/mãos)
  - `Result.dc.html` — resultados lado a lado, integridade, métricas, revisão
  - `Runs.dc.html` — Experimentos: os runs reais de `experiments/runs` (snapshot 24/09/2026)
  - `Setup.dc.html` — Ambiente: máquina atual vs alvo, modelos registrados e autorização do download de 19 GB
  - `canvas.json` — posição das telas no canvas

Os arquivos `.dc.html` dependem do runtime do editor Design (`support.js`), então não
abrem sozinhos no navegador; servem como backup/versão das telas. Se o canvas mudar,
peça para atualizar esta cópia.
