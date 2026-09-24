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

## Benchmark e avaliação

- Benchmark fixo é dado + protocolo + versão de avaliação; manifestos sem mídias
  são apenas vagas planejadas. O check atual deve retornar BLOCKED.
- Detectar corpo não garante âncoras visíveis. No retrato astronaut, corpo/rosto
  são detectados, mas quadris não têm visibilidade suficiente: PCK corporal null
  é correto; não reduzir o threshold para forçar uma nota.
- Sem observações, null; nunca zero de erro como evidência de qualidade.
- PCK inclui falhas de detecção no output como incorretas. Erro condicional deve
  sempre ser lido junto da cobertura.
- Aceleração baixa pode significar um personagem congelado. Avaliar movimento
  e identidade junto da estabilidade, com revisão visual antes de promoção.
- Loader baseline ainda não fixa revisão dos pesos HF; não afirmar reprodução
  exata entre caches diferentes. Ver ADR-007.

## Leitura para continuidade

Ler STATUS.md e HANDOFF.md para comandos, evidências e próximos experimentos.
A pesquisa anterior não foi reauditada nesta sessão; não assumir que testes CPU
validam afirmações de licença, qualidade ou VRAM dessa pesquisa.
