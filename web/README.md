# Site MOVA (local)

O site usa `web/server.py` para consultar o core Python e iniciar extração/inferência local. As seis telas preservam o layout do canvas em `design/canvas/project/`; `dc-runtime.js` renderiza o canvas e `canvas-live.js` conecta seus componentes à API. O CSS adicional fica em `canvas-live.css`.

A identidade visual e as animações da tela Início ficam no canvas fonte `design/canvas/project/Home.dc.html`; o HTML em `web/index.html` é gerado pelo build.

- Iniciar na raiz do projeto: `.venv/Scripts/python web/server.py` e abrir `http://127.0.0.1:8000`.
- Se a página mostrar “Detectando máquina…” permanentemente, confirme que `/api/info` responde JSON. Um servidor estático como `python -m http.server` não fornece a API.
- Páginas: `index.html` (Início), `estudio.html`, `movimento.html`, `resultados.html`, `experimentos.html`, `ambiente.html`.
- Ambiente lê `core.info`; Experimentos lê os `run.json`; Movimento chama `core.preprocess`; Estúdio chama `core.inference`; Resultados exibe os MP4 gerados e registrados.
- O servidor escuta somente em `127.0.0.1`. Uploads ficam em `assets/web_uploads/` e saídas em `outputs/`; ambos são ignorados pelo Git. Arquivos têm limite de 200 MB. Apenas uma execução pesada por vez.
- Wan só usa pesos que já estejam no cache (`allow_download=False`). A interface não baixa modelos. A execução ainda depende de mídia válida, RAM e VRAM suficientes.
- **Não edite os `.html` à mão.** Atualize o canvas em `design/canvas/project/` e rode:

```bash
python scripts/build_web.py
```

`scripts/build_web.py` gera o HTML do canvas e injeta a conexão com a API. Não abra os HTMLs com `file://` nem com `python -m http.server`: essas opções não fornecem a API. Após alterar o servidor, reinicie-o.
