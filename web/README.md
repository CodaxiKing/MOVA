# Site MOVA (local)

O site usa `web/server.py` para consultar o core Python e iniciar extração/inferência local. A interface ativa está em `app.js`/`app.css`; as telas do canvas permanecem como fonte visual em `design/canvas/project/`.

- Iniciar na raiz do projeto: `.venv/Scripts/python web/server.py` e abrir `http://127.0.0.1:8000`.
- Páginas: `index.html` (Início), `estudio.html`, `movimento.html`, `resultados.html`, `experimentos.html`, `ambiente.html`.
- Ambiente lê `core.info`; Experimentos lê os `run.json`; Movimento chama `core.preprocess`; Estúdio chama `core.inference`; Resultados exibe os MP4 gerados e registrados.
- O servidor escuta somente em `127.0.0.1`. Uploads ficam em `assets/web_uploads/` e saídas em `outputs/`; ambos são ignorados pelo Git. Arquivos têm limite de 200 MB. Apenas uma execução pesada por vez.
- Wan só usa pesos que já estejam no cache (`allow_download=False`). A interface não baixa modelos. A execução ainda depende de mídia válida, RAM e VRAM suficientes.
- **Não edite os `.html` à mão.** Atualize o canvas em `design/canvas/project/` e rode:

```bash
python scripts/build_web.py
```

`scripts/build_web.py` ainda gera o HTML do canvas e injeta a interface funcional. Não abra os HTMLs com `file://` nem com `python -m http.server`: essas opções não fornecem a API.
