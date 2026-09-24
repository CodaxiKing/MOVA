# Project Status

## Current Phase
Phase 1 — Baseline (código pronto, execução real bloqueada) · Phase 2 — Motion Extraction (implementada em CPU)

## Overall Progress
[███░░░░░░░] ~25% (Fases 0 e 2 concluídas na parte CPU; Fase 1 aguardando GPU; Fases 3–6 não iniciadas)

## Auditoria e implementação — Codex, 2026-09-24

- Estado inicial: branch main, commit 3174705, árvore limpa.
- CUDA indisponível confirmado; PyTorch 2.14.0+cpu; RAM disponível 0.99/15.69 GB.
- Implementado: validação de pixels antes de conversão e de output.mp4 por
  decodificação completa; relatório em stats.output_validation.
- Verificado: suíte completa, incluindo geração VACE minúscula → encoding → validação.
- Resultado: **35 passed in 12.55s**.
- Comando: `.venv/Scripts/python -m pytest -q -p no:cacheprovider --basetemp outputs/test-final-20260924`.
- Execução fora da restrição de sandbox: tentativas restritas deram PermissionError
  nas pastas temporárias do pytest; não eram falhas funcionais dos componentes.
- Baseline real, qualidade e suporte a 8 GB continuam não verificados.
- A porcentagem e numeração de fases acima são estimativas históricas; não são critérios de PASS do anexo 49.

## Working (verificação anterior)
- Ambiente CPU: Python 3.12.10, PyTorch 2.14.0+cpu, diffusers 0.40.0, MediaPipe 1.0.1
- `scripts/check_env.py` — detecção de GPU/VRAM/RAM e seleção automática de perfil (verificado só no caminho sem CUDA)
- `scripts/check_model_size.py` — tamanho de repositórios HF sem baixar
- Extração de movimento body/face/hands → `.pt` + prévias + controle OpenPose (vídeo sintético "astronaut": corpo 100%, rosto 100%)
- Contrato do baseline com `WanVACEPipeline` (pipeline minúsculo aleatório em CPU)
- Registro automático de experimentos (`experiments/runs/<id>/run.json`)
- Gates de segurança do baseline (sem CUDA → recusa; sem cache → mostra tamanho e recusa download)

## In Progress
- Nada em execução.

## Not Started
- Execução real do baseline (EXP-001)
- Comparação de representações de movimento (EXP-003)
- DWPose vs render MediaPipe (EXP-004)
- Motion Encoder / Projection / Temporal Attention / Motion Adapter
- Identity Encoder
- Dataset experimental, pipeline de treino, losses
- Avaliação automatizada (`evaluation/`)

## Blocked
- **Baseline com pesos reais**: máquina atual sem GPU NVIDIA (Intel Iris Xe). Precisa da máquina com RTX 3060.
- **Download de Wan2.1-VACE-1.3B-diffusers (19.04 GB)**: aguardando autorização do usuário; e a máquina atual só tem 17.9 GB livres.
- **Extração em vídeo real**: `assets/reference/maya.png` e `assets/motion/dance.mp4` não existem — usuário precisa fornecer.

## Last Verified
2026-09-24

## Last Validation anterior (substituída pela auditoria acima)
Command:
```bash
.venv/Scripts/python -m pytest -q
```
Result:
```text
28 passed in 25.57s   (máquina sem GPU; testes GPU não existem ainda)
```
