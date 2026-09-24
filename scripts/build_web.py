"""Gera o site estático em web/ a partir das telas do canvas (design/canvas/project/*.dc.html).

As telas são copiadas sem alterar markup nem lógica; só trocamos o runtime do editor (support.js)
por web/dc-runtime.js e os links *.dc.html pelos nomes das páginas do site.

Uso: python scripts/build_web.py
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "design" / "canvas" / "project"
OUT = ROOT / "web"

PAGES = {
    "Home.dc.html": "index.html",
    "Main.dc.html": "estudio.html",
    "Motion.dc.html": "movimento.html",
    "Result.dc.html": "resultados.html",
    "Runs.dc.html": "experimentos.html",
    "Setup.dc.html": "ambiente.html",
}

TITLES = {
    "index.html": "MOVA | Motion Control",
    "estudio.html": "Estúdio · MOVA",
    "movimento.html": "Movimento · MOVA",
    "resultados.html": "Resultados · MOVA",
    "experimentos.html": "Experimentos · MOVA",
    "ambiente.html": "Ambiente · MOVA",
}

# Únicos desvios do canvas: na Início o título ocupa 4 linhas na coluna de 712 px, o que estoura a
# altura fixa do hero (760 px) e sobrepõe "Ferramentas". Altura mínima em vez de fixa resolve sem
# mudar nada visualmente além de remover a sobreposição.
PATCHES = {
    "Home.dc.html": [
        ('<div style="width: 1440px; height: 1760px;', '<div style="width: 1440px; min-height: 1760px;'),
        ('<section aria-labelledby="hero-h" style="height: 760px;', '<section aria-labelledby="hero-h" style="min-height: 760px;'),
    ],
}

PAGE_CSS = (
    "<style>html,body{background:#0a0a0b}#dc-root>div{margin:0 auto}"
    "input[type=range]{cursor:pointer}</style>"
)


def one(pattern: str, text: str, name: str) -> re.Match[str]:
    m = re.search(pattern, text, re.S)
    if not m:
        raise SystemExit(f"{name}: padrão não encontrado: {pattern}")
    return m


def convert(src: Path, out_name: str) -> str:
    text = src.read_text(encoding="utf-8")
    for old, new in PATCHES.get(src.name, []):
        if old not in text:
            raise SystemExit(f"{src.name}: patch não aplicável (canvas mudou?): {old}")
        text = text.replace(old, new)
    for old, new in PAGES.items():
        text = text.replace(old, new)
    helmet = one(r"<helmet>(.*?)</helmet>", text, src.name).group(1).strip()
    body = one(r"</helmet>(.*?)</x-dc>", text, src.name).group(1).strip()
    script = one(r'(<script type="text/x-dc".*?</script>)', text, src.name).group(1)
    return "\n".join([
        "<!doctype html>",
        '<html lang="pt-BR">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        '<meta name="theme-color" content="#0a0a0b">',
        f"<title>{TITLES[out_name]}</title>",
        f"<!-- Gerado por scripts/build_web.py a partir de design/canvas/project/{src.name}. Não editar à mão. -->",
        helmet,
        PAGE_CSS,
        '<script src="dc-runtime.js" defer></script>',
        "</head>",
        "<body>",
        '<div id="dc-root"></div>',
        '<template id="dc-template">',
        body,
        "</template>",
        script,
        "</body>",
        "</html>",
        "",
    ])


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for src_name, out_name in PAGES.items():
        (OUT / out_name).write_text(convert(SRC / src_name, out_name), encoding="utf-8", newline="\n")
        print(f"{src_name} -> web/{out_name}")


if __name__ == "__main__":
    main()
