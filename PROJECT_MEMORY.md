# Project Memory

## Restrições e decisões

Backbone e baseline seguem ADR-001/002. Download grande exige autorização.
Baseline real deve preceder adapters/treino. Máquina atual sem CUDA;
compatibilidade com RTX 3060 8 GB permanece não medida.

## Descobertas verificadas — 2026-09-24

- Os arrays do modelo devem ser validados antes de cast para uint8: a conversão
  pode esconder NaN/Inf. evaluation/video.py aplica essa regra.
- OpenCV pode encerrar leitura sem distinguir EOF de erro; a validação usa
  contagem esperada e uma segunda decodificação FFmpeg estrita.
- Integridade do arquivo não comprova identidade, transferência ou qualidade.
- Pytest em sandbox deu PermissionError em diretórios temporários; a execução
  autorizada fora da restrição passou (35 testes). Não modificar algoritmos para
  contornar uma falha de permissões do ambiente.

## Continuidade

Ler STATUS.md e HANDOFF.md para comandos, evidências e próximos experimentos.
A pesquisa anterior não foi reauditada nesta sessão; não assumir que testes CPU
validam afirmações de licença, qualidade ou VRAM dessa pesquisa.
