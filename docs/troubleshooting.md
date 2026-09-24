# Troubleshooting

| Sintoma | Causa | Solução |
|---|---|---|
| `nvidia-smi: command not found`, `CUDA available: False` | Máquina sem GPU NVIDIA (ex.: HP ProBook com Intel Iris Xe) ou PyTorch CPU | Usar a máquina da RTX 3060; instalar torch cu12x |
| `inference_baseline.py` sai com "No CUDA GPU detected" | Gate de segurança | Rodar na GPU (ou `--allow-cpu`, horas) |
| "Model ... is not cached. Download size: 19.04 GB" | Gate de download | Confirmar disco e rodar com `--allow-download` |
| Prévia do corpo vazia | (corrigido) coluna z usada como visibilidade | usar `render.body_xyv` |
| Mão esquerda/direita trocadas | MediaPipe assume imagem espelhada | com corpo detectado, atribuição é pelo punho; para selfie use `--set mirrored_input=true` |
| Logs `W0000 ... inference_feedback_manager` / `landmark_projection_calculator` | Avisos internos do MediaPipe | inofensivos |
| OOM de RAM ao codificar prompt | UMT5 bf16 ~11 GB | fechar outros apps; embeddings ficam cacheados depois da primeira vez |
| OOM de VRAM | limite físico | reduzir frames/área, sequential offload — registrar no experimento; **não** é bug |
| `num_frames` alterado automaticamente | Wan exige 4k+1 | esperado |
