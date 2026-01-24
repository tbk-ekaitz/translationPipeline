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
- llama.cpp server compilado

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

### 3. Configurar modelo Kazajo (KazLLM) con llama.cpp

```bash
# Clonar llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Compilar con soporte CUDA (para GPU)
make LLAMA_CUDA=1

# Descargar modelo KazLLM GGUF
wget https://huggingface.co/issai/LLama-3.1-KazLLM-1.0-8B-GGUF4/resolve/main/LLama-3.1-KazLLM-1.0-8B-GGUF4.gguf

# Ejecutar servidor llama.cpp
./server -m LLama-3.1-KazLLM-1.0-8B-GGUF4.gguf -c 4096 --host 0.0.0.0 --port 8080
```

El servidor correrá en `http://localhost:8080`.

**Optimización de cuantización:**
- `GGUF4` (Q4_K_M): Balance calidad/velocidad
- `GGUF6` (Q6_K): Mayor calidad, más VRAM
- `GGUF8` (Q8_0): Calidad máxima

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

### Múltiples instancias de modelos

Para máxima velocidad con 159GB VRAM, puedes ejecutar múltiples instancias:

```bash
# Terminal 1: Ollama para Ruso
ollama run hf.co/t-tech/T-pro-it-2.0-GGUF:Q4_K_M

# Terminal 2: llama.cpp para Kazajo
cd llama.cpp
./server -m KazLLM.gguf -c 4096 --port 8080

# Terminal 3: Pipeline
python pipeline.py documento.docx
```

### Ajustar URLs en config.yaml

```yaml
llm:
  russian:
    base_url: "http://localhost:11434"  # Ollama
  kazakh:
    base_url: "http://localhost:8080"   # llama.cpp
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
- Verifica el modelo: `ollama run hf.co/t-tech/T-pro-it-2.0-GGUF:Q4_K_M`

### Error: "llama.cpp API error"
- Verifica que el servidor esté activo: `curl http://localhost:8080/health`
- Revisa logs del servidor llama.cpp

### Error: "Out of memory"
- Usa cuantizaciones más agresivas (Q4 en lugar de Q6)
- Reduce el contexto en llama.cpp: `-c 2048`

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
