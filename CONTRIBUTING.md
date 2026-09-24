# Contributing

1. Leia `AGENTS.md` e `CLAUDE.md`.
2. Um componente por módulo; configs em `configs/*.yaml`; caminhos relativos via `common.config.resolve_path`.
3. Código tipado quando possível; comentários só quando o "porquê" não é óbvio.
4. Todo código novo com teste em `tests/` que rode em CPU (use modelos minúsculos aleatórios para pipelines de difusão).
5. `pytest -q` deve passar antes de commitar.
6. Commits pequenos e lógicos (`feat(scope): ...`, `fix(scope): ...`, `docs: ...`, `exp: ...`).
7. Nunca commitar pesos, vídeos ou imagens.
8. Experimentos: `ExperimentRun` grava `run.json`; escreva o relatório em `docs/experiments/EXP-XXX.md`.
9. Mudanças arquiteturais exigem ADR em `DECISIONS.md`.
10. Use apenas modelos, código e dados com licença que permita o uso pretendido; registre a licença em `docs/research/sources.md`.
