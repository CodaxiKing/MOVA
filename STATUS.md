# Project Status

## Estado atual — 2026-09-24

- Baseline: implementado, validado apenas com pipeline minúsculo aleatório; geração real BLOQUEADA.
- Extração MediaPipe: implementada e testada em CPU.
- Avaliação: integridade e métricas de movimento implementadas e testadas em CPU.
- Benchmark: protocolo de 20 vagas e ferramentas implementados; mídias reais ausentes, conjunto não congelado.
- Adapter, identidade treinável e treinamento: não iniciados; MVP e equivalência ao Kling não demonstrados.

## Entrega desta sessão

- Auditoria inicial: branch main, HEAD 6943f0f, árvore limpa; alterações anteriores preservadas.
- `evaluation/motion.py`: PCK/erro de corpo e mãos, cobertura, expressão e aceleração, sem interpolação de ausências.
- `evaluation/protocol.py`: manifesto, validação e lock SHA-256 de mídia/condições.
- `evaluation/benchmark.py`: re-extração, relatório por caso, registro automático e comparação estrita.
- `scripts/benchmark.py`: check, prepare, freeze, generate, evaluate e compare.
- `benchmark/v1.draft.yaml`: 20 vagas em dez categorias; templates de revisão visual e registro de falhas.
- ADR-007 e EXP-005 documentam decisões, limitações e próximos experimentos.

## Verificação executada

```bash
.venv/Scripts/python scripts/check_env.py --cuda-test
.venv/Scripts/python -m pytest -q -p no:cacheprovider --basetemp outputs/test-benchmark-release-20260924
.venv/Scripts/python scripts/benchmark.py check
```

- Ambiente: Python 3.12.10, PyTorch 2.14.0+cpu, CUDA False; RAM disponível 0.94/15.69 GB; disco C livre 16.66 GB.
- Testes: **51 passed in 29.42s**. Execução com acesso fora do sandbox às pastas temporárias, devido às restrições já documentadas.
- Testes cobrem erros conhecidos de keypoints, ausências, alinhamento, adulteração de dados/lock, integridade, extração real MediaPipe e repetição de avaliação facial.
- Retrato astronaut: corpo/rosto detectados, mas quadris insuficientemente visíveis; PCK corporal null é esperado. Comparação facial do vídeo consigo mesmo teve erro zero e foi repetida. Não é vídeo gerado por difusão.
- Orquestração generate testada com baseline simulado; geração CUDA real permanece não testada.
- Check: código de saída 2, **BLOCKED**, 20 casos, 140 pendências de arquivos/metadados. Relatório local: outputs/benchmark-preflight.json.
- A mensagem de decoder sobre moov ausente durante pytest vem do teste de arquivo inválido; suíte terminou com código 0.

## Bloqueios reais

- Máquina atual sem CUDA; precisa executar na RTX 3060.
- Pesos Wan não presentes no cache; download estimado anteriormente em 19.04 GB segue sem autorização e sem espaço suficiente nesta máquina.
- Imagens/vídeos e fontes/licenças não fornecidos; nenhum dataset novo baixado.

## Limitações

- PCK normalizado não mede trajetória global; expressão não mede identidade.
- Aceleração não mede flicker de textura. Não há score global nem promoção automática.
- Identidade, head pose, fluxo óptico e qualidade perceptual continuam pendentes.
- Loader legado não fixa revisão HF: geração registra model_revision=null.
- Compatibilidade com 8 GB, qualidade, tempo de geração real e ganho sobre baseline não medidos.

## Próxima ação

Preencher as mídias/metadados conforme benchmark/README.md. Na máquina CUDA,
executar primeiro EXP-001, depois congelar o benchmark e gerar duas execuções
comparáveis; avaliar e revisar antes de iniciar adapters ou treinamento.
