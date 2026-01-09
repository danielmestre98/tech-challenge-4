# Tech Challenge – Fase 4 (Análise de Vídeo)

Este projeto processa um vídeo e entrega:

- **Reconhecimento facial** (detecção + ID consistente por pessoa ao longo do vídeo)
- **Análise de expressões emocionais** (por pessoa e ao longo do tempo)
- **Detecção/categorização de atividades** (modelo pré-treinado de action recognition)
- **Geração de resumo** (relatório texto com métricas e principais achados)
- **Anomalias** (movimento fora do padrão: gestos bruscos/comportamentos atípicos)

## Requisitos

- Python 3.10+ (recomendado 3.11)
- GPU (CUDA) **opcional**, mas recomendado
- Um vídeo `.mp4`

## Instalação (recomendado)

1. Crie e ative um ambiente virtual.

2. Instale **PyTorch + TorchVision** conforme sua versão de CUDA:

- Veja as instruções oficiais do PyTorch (opção mais segura).

3. Instale as dependências do projeto:

```bash
pip install -r requirements.txt
```

4. (Opcional, para acelerar o InsightFace na GPU)
   Instale o runtime ONNX para GPU:

```bash
pip install onnxruntime-gpu
```

> Observação: o `insightface` baixa modelos automaticamente na primeira execução e guarda cache no diretório do usuário.
> Se quiser controlar o cache: `export INSIGHTFACE_HOME=/caminho/para/cache`.

## Como rodar

Exemplo básico:

```bash
python -m src.main --video /caminho/video.mp4 --outdir outputs/run1
```

Com GPU nos modelos PyTorch (emoção/atividade):

```bash
python -m src.main --video video.mp4 --outdir outputs/run1 --device cuda
```

Parâmetros úteis:

- `--frame-skip 1` processa todos os frames; `--frame-skip 2` processa 1 a cada 2 frames (mais rápido)
- `--max-frames 1000` limita a execução (debug)
- `--emotion-model trpakov/vit-face-expression` (modelo do Hugging Face; pode trocar)
- `--activity-window 16` tamanho do clip (frames) para atividade
- `--activity-stride 8` a cada quantos frames recalcular atividade

## Saídas geradas

No `--outdir`:

- `annotated.mp4` – vídeo com caixas/labels
- `events.jsonl` – log por frame (faces/emotions) e eventos (atividade/anomalia)
- `report.txt` – resumo final (inclui: frames analisados + número de anomalias)
