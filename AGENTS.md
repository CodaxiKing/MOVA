# AGENTS.md — Instruções para qualquer agente de IA

## Ordem de leitura obrigatória (antes de modificar qualquer coisa)

1. `AGENTS.md` (este arquivo)
2. `CLAUDE.md` — manual do projeto
3. `STATUS.md` — estado real
4. `HANDOFF.md` — onde a última sessão parou
5. `TODO.md`
6. `DECISIONS.md` — antes de alterar qualquer decisão arquitetural
7. Documentação relevante em `docs/` antes de modificar um componente

Nunca assuma o estado do projeto sem verificar documentação **e** código. Em caso de dúvida, rode:

```bash
.venv/Scripts/python scripts/check_env.py --cuda-test
.venv/Scripts/python -m pytest -q
git log --oneline -15
```

## Resumo de estado (produza internamente no início da sessão)

```text
PROJECT STATE
-------------
Current phase:
Current branch:
Last completed task:
Current task:
Current blocker:
Last successful test:
Next action:
```

Não modifique código antes de entender esse estado. Só pergunte ao usuário quando a decisão exigir intervenção humana (ex.: autorizar download grande, licença, escolher entre alternativas equivalentes).

## Regras invioláveis

- Não inventar resultados; não marcar DONE/PASS sem executar.
- Não baixar dezenas de GB sem autorização (informar tamanho, licença, disco, VRAM, motivo).
- Não copiar código/pesos proprietários (Kling etc.). Não afirmar que o MOVA reproduz arquitetura proprietária.
- Não versionar `*.safetensors *.ckpt *.pth *.pt *.mp4 *.png`.
- Não substituir o backbone sem novo ADR + atualização de `CLAUDE.md` e `docs/architecture.md`.
- Não mascarar falta de VRAM como bug de código.

## Sincronização da documentação

```text
Código alterado → Teste executado → Resultado registrado → STATUS.md → TODO.md → CHANGELOG.md → HANDOFF.md
Decisão arquitetural → DECISIONS.md → CLAUDE.md → docs/architecture.md
```

## Checklist de fim de sessão

```text
[ ] Código salvo e commitado
[ ] Testes executados (pytest -q)
[ ] STATUS.md atualizado
[ ] TODO.md atualizado
[ ] CHANGELOG.md atualizado (se necessário)
[ ] DECISIONS.md atualizado (se necessário)
[ ] HANDOFF.md atualizado
[ ] CLAUDE.md atualizado (se necessário)
[ ] README.md atualizado (se necessário)
```

## Requisito de continuidade

"Um novo agente sem acesso ao histórico da conversa deve conseguir continuar o projeto apenas lendo o repositório."
