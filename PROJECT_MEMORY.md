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

## Arquitetura em camadas (ADR-009) — descobertas verificadas 2026-09-24

- O caminho de geração é determinístico em CPU: o VACE minúsculo com semente dá o mesmo SHA-256 em repetições;
  isso permite provar refatorações bit a bit (`benchmark/regression`). Qualquer mudança que altere esse hash
  mudou a numérica — investigar antes de seguir.
- `torch.no_grad` (runtime) e o `no_grad` interno do pipeline não alteram a saída; o gerador precisa continuar
  sendo de CPU para manter o ruído inicial igual entre devices.
- Na CPU com torch 2.14, fp16 e bf16 rodam o VACE minúsculo com saída finita, apesar do aviso do Diffusers de
  que fp16 em CPU "vai falhar". Isso vale para o modelo minúsculo; o 1.3B em CPU não foi testado.
- `gc.collect()` no unload custa ~0.14 s num processo com torch/diffusers carregados: é o único overhead
  medido da nova arquitetura.
- `argparse.REMAINDER` em subcomando descarta flags iniciais (`mova test -q`); comandos de repasse são tratados
  antes do argparse.
- No Windows, `Path.write_text` grava CRLF; o repositório é LF (`.gitattributes`). Usar `write_bytes` ou `newline`.

## Qualidade do motion control — descobertas verificadas 2026-09-24

- Identidade: 24 landmarks rígidos do Face Mesh bastam para detectar deriva de morfologia (0.009 → 0.055 com rosto
  29 % mais largo), mas não distinguem pessoas de proporções parecidas. Cor Lab separa roupa/pele trocada.
- Temporal: warp error distingue textura suave (0.0009) de ruído por frame (0.05) e flicker (0.157); vídeo congelado
  dá warp 0, então sempre ler junto com o fluxo médio e a razão contra o driver.
- Retargeting: usar comprimentos 3D (world) da referência e multiplicar o vetor 2D do ator pelo mesmo fator
  preserva o escorço; o caminho da raiz precisa escalar com o torso (um teste pegou esse erro).
- VACE prefixa a referência como frame latente extra; o forward do transformer VACE exige controle.
- One-Euro em dança rápida a 16 fps aumenta o erro mais do que reduz o jitter; padrão sem suavização.
- rot6d de todos os ossos é muito sensível a ruído (ossos curtos); só ossos longos: invariante e robusto (sintético).
- Licenças: pesos InsightFace são não comerciais e o LivePortrait os embute; UniAnimate-DiT não tem licença;
  AIST Dance DB é só pesquisa acadêmica.
- No bash desta máquina, heredocs com aspas misturadas quebram; escrever scripts de edição com a ferramenta Write.

## Leitura para continuidade

Ler STATUS.md e HANDOFF.md para comandos, evidências e próximos experimentos.
A pesquisa anterior não foi reauditada nesta sessão; não assumir que testes CPU
validam afirmações de licença, qualidade ou VRAM dessa pesquisa.
