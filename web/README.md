# Site MOVA (protótipo)

Site estático gerado a partir do canvas de design (`design/canvas/project/*.dc.html`).
**É uma simulação: nenhum modelo roda aqui.**

- Abrir: dê dois cliques em `index.html`, ou sirva a pasta com `python -m http.server -d web 8000`.
- Páginas: `index.html` (Início), `estudio.html`, `movimento.html`, `resultados.html`, `experimentos.html`, `ambiente.html`.
- `dc-runtime.js` substitui o runtime do editor de design (`{{expr}}`, `<sc-for>`, `<sc-if>`, `onClick`/`onChange`).
- **Não edite os `.html` à mão.** Atualize o canvas em `design/canvas/project/` e rode:

```bash
python scripts/build_web.py
```

O único desvio do canvas fica em `PATCHES`, no script: na Início, o hero usa altura mínima em vez de fixa,
porque o título de 4 linhas cobria "Ferramentas".
