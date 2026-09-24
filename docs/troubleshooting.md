# Troubleshooting

| Sintoma | Causa | Solução |
|---|---|---|
| `nvidia-smi: command not found`, `CUDA available: False` | Máquina sem GPU NVIDIA (ex.: HP ProBook com Intel Iris Xe) ou PyTorch CPU | Usar a máquina da RTX 3060; instalar torch cu12x |
| `ERROR [device_not_supported]: Device cpu for model 'wan2.1-vace-1.3b' …` | Gate de segurança (1.3B em CPU leva horas) | Rodar na GPU (ou `--allow-cpu`) |
| `ERROR [model_weights_missing]: Missing 19.04 GB …` | Gate de download | Confirmar disco e rodar com `--allow-download` |
| Prévia do corpo vazia | (corrigido) coluna z usada como visibilidade | usar `render.body_xyv` |
| Mão esquerda/direita trocadas | MediaPipe assume imagem espelhada | com corpo detectado, atribuição é pelo punho; para selfie use `--set mirrored_input=true` |
| Logs `W0000 ... inference_feedback_manager` / `landmark_projection_calculator` | Avisos internos do MediaPipe | inofensivos |
| OOM de RAM ao codificar prompt | UMT5 bf16 ~11 GB | fechar outros apps; embeddings ficam cacheados depois da primeira vez |
| OOM de VRAM | limite físico | reduzir frames/área, sequential offload — registrar no experimento; **não** é bug |
| `num_frames` alterado automaticamente | Wan exige 4k+1 | esperado |

## Códigos de erro (`mova`, `scripts/inference_baseline.py`; ADR-009)

Todos saem antes de carregar pesos, com exit code 2 e uma dica.

| Código | Quando | O que fazer |
|---|---|---|
| `model_not_found` | nome/alias fora do registry | `mova info` lista os modelos |
| `runtime_not_available` | runtime desconhecido, não implementado (onnx, tensorrt) ou pacote do modelo ausente | `--runtime pytorch`; `pip install -r requirements.txt` |
| `model_runtime_incompatible` | o modelo não declara suporte ao runtime | ver `mova info --model <m>` |
| `device_not_supported` | device inexistente (`cuda:1` sem 2ª GPU), string inválida (`mps`, `tpu`) ou CPU sem `--allow-cpu` | `--device auto`, conferir driver/torch CUDA |
| `precision_not_supported` | precisão desconhecida ou não suportada pelo device/modelo (ex.: bf16 em GPU pré-Ampere) | `--precision auto` |
| `invalid_config` | config inexistente, chave desconhecida em `runtime:`, offload em CPU, `--model` conflitando com `--config` | corrigir config/flags |
| `invalid_input` | referência/vídeo ausentes, `--output` já existe ou não é `.mp4` | corrigir caminho ou `--overwrite` |
| `model_weights_missing` | pesos ausentes/incompletos sem `--allow-download`, manifesto não resolvido offline | ver seção de reprodutibilidade em `docs/inference.md` |
| `insufficient_resources` / `insufficient_vram` | pré-checagem de disco/RAM/VRAM recusou | liberar recursos, reduzir frames/área; `--skip-resource-check` só se a estimativa estiver errada |
