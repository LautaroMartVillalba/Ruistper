#!/bin/sh
# Entrypoint — descarga el modelo GGML si no está presente, luego ejecuta el binario.
set -e

MODEL_FILE="${MODELS_DIR:-/app/models}/ggml-${WHISPER_MODEL:-base}.bin"

if [ ! -f "$MODEL_FILE" ]; then
    echo "⬇️  Modelo no encontrado en $MODEL_FILE"
    echo "⬇️  Descargando ggml-${WHISPER_MODEL}.bin desde Hugging Face..."
    mkdir -p "$(dirname "$MODEL_FILE")"
    curl -L --fail --show-error --progress-bar \
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-${WHISPER_MODEL}.bin" \
        -o "$MODEL_FILE" \
    && echo "✅ Modelo descargado: $MODEL_FILE" \
    || { echo "❌ Error al descargar el modelo. Verifica WHISPER_MODEL y la conexión."; exit 1; }
fi

exec "$@"
