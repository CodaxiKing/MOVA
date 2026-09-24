# MOVA — Motion Control de Personagens (open-source, low-VRAM)

```text
reference.png + motion.mp4 (+ imagens extras de identidade)  ─►  MOVA  ─►  personagem executando o movimento.mp4
```

Objetivo: animar uma personagem a partir de um vídeo de movimento, preservando identidade, rosto, cabelo, roupa, proporções, mãos, expressão e consistência temporal — rodando numa **RTX 3060 8 GB**, com backbone pré-treinado congelado e módulos pequenos treináveis.

> Inspirado em princípios técnicos publicamente documentados (ex.: Kling-MotionControl Technical Report). **Não** reproduz nem copia arquitetura, código ou pesos proprietários.

## Status

Ver [STATUS.md](STATUS.md). Resumo: pesquisa concluída; extração de movimento funcionando em CPU; baseline (Wan2.1-VACE-1.3B) implementado mas ainda não executado com pesos reais (precisa da GPU).

## Quickstart

```bash
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu128   # GPU
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python scripts/check_env.py --cuda-test
.venv/Scripts/python -m pytest -q
```

Extrair movimento (CPU):

```bash
.venv/Scripts/python scripts/extract_motion.py --video assets/motion/dance.mp4 --out outputs/motion/dance
```

Baseline (GPU; baixa 19 GB apenas com `--allow-download`):

```bash
.venv/Scripts/python scripts/check_model_size.py Wan-AI/Wan2.1-VACE-1.3B-diffusers
.venv/Scripts/python scripts/inference_baseline.py --allow-download
```

## Documentação

| Arquivo | Conteúdo |
|---|---|
| [CLAUDE.md](CLAUDE.md) / [AGENTS.md](AGENTS.md) | Manual para agentes de IA |
| [HANDOFF.md](HANDOFF.md) | Onde a última sessão parou |
| [DECISIONS.md](DECISIONS.md) | ADRs |
| [docs/architecture.md](docs/architecture.md) | Arquitetura atual e planejada |
| [docs/pipeline.md](docs/pipeline.md) | Formato dos dados de movimento |
| [docs/inference.md](docs/inference.md) | Baseline e memória |
| [docs/training.md](docs/training.md) | Plano de treino e losses |
| [docs/dataset.md](docs/dataset.md) | Datasets e licenças |
| [docs/evaluation.md](docs/evaluation.md) | Métricas |
| [docs/research/](docs/research/) | Pesquisa (fontes, backbone, Kling, Wan...) |

## Licença

Apache-2.0 (provisória — ver ADR-006). Modelos de terceiros seguem suas próprias licenças.
