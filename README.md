# Translation Pipeline

Pipeline de conversión y traducción de documentos DOCX/PDF a múltiples idiomas (Ruso y Kazajo).

## Arquitectura del Pipeline

```
{DOCX/PDF}
    ↓
[markitdown] → documento.md
    ↓
[Segmentador] → Lista ordenada de frases
    ↓
[Batch Processor - Fase 1] → Todos los batches RUSOS (GPU saturada)
    ↓
[Batch Processor - Fase 2] → Todos los batches KAZAJOS (GPU saturada)
    ↓
[Output Generator] → JSON/TXT formateado (orden preservado)
    ↓
{Archivo de salida}
```

**Batching secuencial:** Primero completa TODAS las traducciones rusas en batches, luego TODAS las kazajas. Cada traducción se asigna a su frase correcta usando índices explícitos.

## Características

- ✅ Conversión automática DOCX/PDF → Markdown
- ✅ Segmentación inteligente por frases
- ✅ **Batch processing para máxima GPU utilization** (5-8x más rápido)
- ✅ Traducción secuencial a Ruso y Kazajo (sin complicaciones)
- ✅ Preserva orden exacto de frases con índices
- ✅ Output JSON con formato bonito (pretty-print)
- ✅ Output TXT legible para copy-paste
- ✅ Soporte para modelos cuantizados (GGUF)
- ✅ Batch sizes configurables por modelo

## Requisitos

- Python 3.8+
- 159GB VRAM (para cargar ambos modelos simultáneamente)
- Ollama instalado

## Instalación

### 1. Instalar dependencias Python

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"
```

### 2. Configurar modelo Ruso (T-pro-it-2.0) con Ollama

```bash
# Instalar Ollama si no lo tienes
curl -fsSL https://ollama.com/install.sh | sh

# Descargar y ejecutar T-pro-it-2.0
ollama run hf.co/t-tech/T-pro-it-2.0-GGUF:Q4_K_M
```

Ollama correrá en `http://localhost:11434` por defecto.

**Modelos alternativos para Ruso:**
- `hf.co/t-tech/T-pro-it-2.1-GGUF:Q4_K_M` (versión mejorada)
- `hf.co/t-tech/T-pro-it-2.0-GGUF:Q6_K` (mayor calidad, más VRAM)

### 3. Configurar modelo Kazajo (KazLLM) con Ollama

```bash
# Descargar modelo KazLLM GGUF desde HuggingFace
wget https://huggingface.co/issai/LLama-3.1-KazLLM-1.0-8B-GGUF4/resolve/main/LLama-3.1-KazLLM-1.0-8B-GGUF4.gguf
# Renombrar para simplicidad
mv LLama-3.1-KazLLM-1.0-8B-GGUF4.gguf KazLLM.gguf

# Crear Modelfile
cat > Modelfile <<EOF
FROM KazLLM.gguf
PARAMETER temperature 0.7
PARAMETER num_ctx 8192
EOF

# Crear modelo en Ollama
ollama create kazllm-local -f Modelfile

# Verificar que funcione
ollama run kazllm-local
```

**Configuración de `num_ctx` (ventana de contexto):**
- `num_ctx 4096`: Frases cortas (<50 palabras)
- `num_ctx 8192`: **Recomendado para frases largas** (hasta ~300 palabras)
- `num_ctx 16384`: Frases muy largas o documentos técnicos complejos
- **Máximo modelo:** 131072 (no recomendado, desperdicia VRAM)

Para frases muy largas, usa `num_ctx 8192` o `16384` para evitar truncamiento.

### 4. Configurar prompts

Edita `config.yaml` y reemplaza los placeholders con tus prompts reales:

```yaml
prompts:
  russian: |
    You are a professional English to Russian translator.
    Translate the following text accurately:
    {text}

  kazakh: |
    You are a professional English to Kazakh translator.
    Translate the following text accurately:
    {text}
```

## Uso

### Ejecución básica

```bash
# Traducir un documento DOCX
python pipeline.py input/documento.docx

# Traducir un PDF
python pipeline.py input/documento.pdf

# Especificar nombre de salida
python pipeline.py input/documento.docx -o mi_traduccion

# Usar archivo de configuración custom
python pipeline.py input/documento.docx -c config.custom.yaml

# Si ya tienes un .md, saltar conversión
python pipeline.py input/documento.md --skip-conversion
```

### Outputs generados

El pipeline genera dos archivos en `./output/`:

1. **JSON** (`documento_translated.json`):
```json
[
  {
    "id": 1,
    "original": "This is the first sentence.",
    "russian": "Это первое предложение.",
    "kazakh": "Бұл бірінші сөйлем."
  },
  {
    "id": 2,
    "original": "This is the second sentence.",
    "russian": "Это второе предложение.",
    "kazakh": "Бұл екінші сөйлем."
  }
]
```

2. **TXT** (`documento_translated.txt`):
```
[1]
Original:  This is the first sentence.
Russian:   Это первое предложение.
Kazakh:    Бұл бірінші сөйлем.

================================================================================

[2]
Original:  This is the second sentence.
Russian:   Это второе предложение.
Kazakh:    Бұл екінші сөйлем.

================================================================================
```

## Estructura de datos

Cada frase se representa como un objeto `Sentence`:

```python
sentence = Sentence(
    id=1,
    text="Original text",
    translations={
        'russian': 'Перевод на русский',
        'kazakh': 'Қазақ тіліне аударма'
    }
)

# Acceso a traducciones
sentence.translations['russian']  # 'Перевод на русский'
sentence.translations['kazakh']   # 'Қазақ тіліне аударма'
```

## Configuración avanzada

### Ejecución con Ollama

Ambos modelos se ejecutan en Ollama (puerto 11434 por defecto):

```bash
# Terminal 1: Mantén Ollama corriendo en background
# (Ollama maneja ambos modelos automáticamente)

# Terminal 2: Ejecutar pipeline
python pipeline.py input/documento.docx
```

**Nota:** Ollama carga modelos bajo demanda. El primer batch puede ser más lento mientras carga el modelo en GPU.

### Configuración en config.yaml

Ambos modelos usan Ollama:

```yaml
llm:
  russian:
    type: "ollama"
    model_name: "hf.co/t-tech/T-pro-it-2.0-GGUF:Q4_K_M"
    base_url: "http://localhost:11434"
    batch_size: 32

  kazakh:
    type: "ollama"
    model_name: "kazllm-local"
    base_url: "http://localhost:11434"  # Mismo servidor Ollama
    batch_size: 48
```

### Optimizar Batch Sizes

El pipeline usa **batch processing secuencial** para saturar la GPU:

**Flujo de batching:**
1. Todas las traducciones al Ruso (en batches de 32)
2. Todas las traducciones al Kazajo (en batches de 48)
3. Orden de frases preservado con índices

**Ajustar batch sizes en `config.yaml`:**

```yaml
llm:
  russian:
    batch_size: 32  # ↓ Menor para modelos grandes (~13B+)
                    # ↑ Mayor si tienes VRAM de sobra
  kazakh:
    batch_size: 48  # KazLLM-8B es más pequeño → batch mayor
```

**Recomendaciones por VRAM:**

| VRAM Total | Russian Batch | Kazakh Batch | Cuantización |
|------------|---------------|--------------|--------------|
| 24GB       | 8-16          | 16-24        | Q4_K_M       |
| 48GB       | 16-24         | 24-40        | Q4_K_M/Q6_K  |
| 80GB       | 24-32         | 40-64        | Q6_K         |
| 159GB      | 32-48         | 48-80        | Q6_K/Q8_0    |

**Ventajas del batching secuencial:**
- ✅ GPU saturada (no espera entre frases)
- ✅ 5-8x más rápido vs frase por frase
- ✅ Sin complicaciones de paralelización
- ✅ Orden garantizado (usa índices explícitos)
- ✅ Batch sizes diferentes por modelo

## Troubleshooting

### Error: "Ollama API error"
- Verifica que Ollama esté corriendo: `ollama list`
- Verifica los modelos:
  ```bash
  ollama run hf.co/t-tech/T-pro-it-2.0-GGUF:Q4_K_M
  ollama run kazllm-local
  ```
- Revisa logs de Ollama: `ollama logs`

### Error: "Out of memory"
- Usa cuantizaciones más agresivas (Q4 en lugar de Q6/Q8)
- Reduce `num_ctx` en Modelfile de kazllm-local: `num_ctx 4096` o `2048`
- Reduce batch sizes en `config.yaml`

### Traducciones de baja calidad
- Usa cuantizaciones mayores (Q6_K, Q8_0)
- Ajusta los prompts en `config.yaml`
- Incrementa `n_predict` en `llm_client.py`

## Estructura del proyecto

```
translationPipeline/
├── src/
│   ├── __init__.py
│   ├── document_converter.py      # DOCX/PDF → Markdown
│   ├── sentence_segmenter.py      # Markdown → Frases
│   ├── llm_client.py              # Clientes Ollama/llama.cpp
│   ├── translation_orchestrator.py # Coordinador de traducciones
│   └── output_generator.py        # JSON/TXT generator
├── input/                         # Documentos de entrada
├── output/                        # Archivos traducidos
├── pipeline.py                    # Script principal
├── config.yaml                    # Configuración
├── requirements.txt
└── README.md
```

## Referencias

- [T-pro-it-2.0 GGUF](https://huggingface.co/t-tech/T-pro-it-2.0-GGUF)
- [KazLLM Models](https://huggingface.co/issai/LLama-3.1-KazLLM-1.0-8B-GGUF4)
- [Ollama Documentation](https://github.com/ollama/ollama)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [MarkItDown](https://github.com/microsoft/markitdown)

## Licencia

MIT
