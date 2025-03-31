<h1 align="center">OpenImageTokenizer 🖼️→🔢</h1>

<div align="center">

**Una interfaz Python elegante para los tokenizadores visuales de SEED-Voken**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-APACHE2.0-green.svg)](LICENSE)

</div>

## 📝 Descripción

OpenImageTokenizer es una biblioteca de Python que proporciona una interfaz simplificada y accesible para los potentes tokenizadores visuales desarrollados por TencentARC en su proyecto [SEED-Voken](https://github.com/TencentARC/SEED-Voken). Este paquete facilita el uso de modelos avanzados como Open-MAGVIT2 e IBQ sin necesidad de configuraciones complejas o entornos de desarrollo especializados.

Similar a cómo los tokenizadores de texto convierten texto en tokens discretos, los tokenizadores visuales convierten imágenes en representaciones discretas (tokens) que pueden ser utilizadas para diversos fines, desde compresión hasta generación de imágenes autorregresiva.

<p align="center">
<img src="https://raw.githubusercontent.com/TencentARC/SEED-Voken/main/assets/comparsion.png" width=90%>
<br><small><i>Comparación de diferentes tokenizadores visuales (imagen de SEED-Voken)</i></small>
</p>

## ✨ Características

- **Interfaz simplificada**: API intuitiva para usar tokenizadores visuales sin necesidad de entender su complejidad interna
- **Descarga automática**: Gestión transparente de checkpoints desde Hugging Face sin intervención manual
- **Configuraciones integradas**: No requiere archivos YAML o JSON externos
- **Visualización de tokens**: Herramientas para visualizar y entender los tokens generados
- **Compatible con múltiples modelos**: Soporte para diferentes versiones de Open-MAGVIT2 e IBQ
- **Multi-plataforma**: Funciona en CPU y GPU sin configuraciones especiales

## 📊 Modelos Soportados

OpenImageTokenizer proporciona acceso a los siguientes modelos avanzados de SEED-Voken:

### Open-MAGVIT2

Tokenizador visual estado del arte con rendimiento superior (`0.39 rFID` para downsampling 8x).

- TencentARC/Open-MAGVIT2-Tokenizer-128-resolution
- TencentARC/Open-MAGVIT2-Tokenizer-256-resolution
- TencentARC/Open-MAGVIT2-Tokenizer-16384-Pretrain
- TencentARC/Open-MAGVIT2-Tokenizer-262144-Pretrain

### IBQ

Tokenizador visual escalable con alta dimensión de código y alta utilización.

- TencentARC/IBQ-Tokenizer-16384
- TencentARC/IBQ-Tokenizer-32768

## 🛠️ Instalación

```bash
pypi no soportado
```

O directamente desde el repositorio:

```bash
git clone https://github.com/F4k3r22/OpenImageTokenizer.git
cd OpenImageTokenizer
pip install -e .
```

## 🚀 Uso Rápido

### Ejemplo Básico

```python
from OpenImageTokenizer import MAGVIT2ImageTokenizer

# Inicializar tokenizador (descarga automática de checkpoints)
tokenizer = MAGVIT2ImageTokenizer("TencentARC/Open-MAGVIT2-Tokenizer-256-resolution")

# Tokenizar una imagen
encoded = tokenizer.encode("ruta/a/imagen.jpg")
tokens = encoded['indices']

# Reconstruir la imagen desde los tokens
reconstructed = tokenizer.decode(encoded['quant'])

# Visualizar los tokens
tokenizer.visualize_tokens(tokens, save_path="tokens_visualization.png")
```

### Procesamiento Completo

```python
# Codificar, decodificar y visualizar en un solo paso
results = tokenizer.process_image("ruta/a/imagen.jpg", "directorio/salida")

print(f"Imagen original: {results['original']}")
print(f"Imagen reconstruida: {results['reconstructed']}")
print(f"Visualización de tokens: {results['tokens']}")
```

## 🔍 Aplicaciones

Los tokenizadores visuales tienen múltiples aplicaciones en visión por computadora e IA:

- **Generación autorregresiva de imágenes**: Base para modelos tipo GPT pero para imágenes
- **Modelos multimodales**: Punto de conexión entre modelos de lenguaje y contenido visual
- **Compresión de imágenes**: Representación eficiente mediante tokens discretos
- **Edición semántica**: Manipulación a nivel de tokens para edición controlada
- **Investigación en generación visual**: Experimentación con diferentes arquitecturas

## 🧩 Componentes Principales

- **MAGVIT2ImageTokenizer**: Clase principal para tokenización con Open-MAGVIT2
- **hf_utils**: Módulo para gestionar la descarga de modelos desde Hugging Face
- **configs**: Configuraciones integradas para los diferentes modelos
- **visualize_tokens**: Utilidades para visualizar y comprender los tokens generados

## 📑 Ejemplo de Script Completo

```python
import os
from OpenImageTokenizer import MAGVIT2ImageTokenizer

# Inicializar tokenizador
tokenizer = MAGVIT2ImageTokenizer("TencentARC/Open-MAGVIT2-Tokenizer-256-resolution")

# Cargar el modelo
tokenizer.load_model()

# Procesar imagen (codificar, visualizar, reconstruir)
image_path = "mi_imagen.jpg"
output_dir = "resultados"

results = tokenizer.process_image(image_path, output_dir)

# Mostrar información sobre los tokens
token_shape = results["token_shape"]
print(f"Forma de los tokens: {token_shape}")
print(f"Total de tokens en la imagen: {token_shape[0] * token_shape[1]}")

print("Archivos generados:")
print(f"  Original: {results['original']}")
print(f"  Reconstruido: {results['reconstructed']}")
print(f"  Visualización de tokens: {results['tokens']}")
```

## 📚 Citas

Si utilizas OpenImageTokenizer en tu investigación, considera citar los trabajos originales:

Para Open-MAGVIT2:

```bibtex
@article{luo2024open,
  title={Open-MAGVIT2: An Open-Source Project Toward Democratizing Auto-regressive Visual Generation},
  author={Luo, Zhuoyan and Shi, Fengyuan and Ge, Yixiao and Yang, Yujiu and Wang, Limin and Shan, Ying},
  journal={arXiv preprint arXiv:2409.04410},
  year={2024}
}
```

Para IBQ:

```bibtex
@article{shi2024taming,
  title={Taming Scalable Visual Tokenizer for Autoregressive Image Generation},
  author={Shi, Fengyuan and Luo, Zhuoyan and Ge, Yixiao and Yang, Yujiu and Shan, Ying and Wang, Limin},
  journal={arXiv preprint arXiv:2412.02692},
  year={2024}
}
```

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Para contribuir:

1. Haz un fork del repositorio
2. Crea una nueva rama (`git checkout -b feature/nueva-funcionalidad`)
3. Haz tus cambios y commitealos (`git commit -m 'Añade nueva funcionalidad'`)
4. Haz push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está licenciado bajo la licencia APACHE 2.0 - consulta el archivo [LICENSE](LICENSE) para más detalles.

## ❤️ Agradecimientos

- [TencentARC](https://github.com/TencentARC) por desarrollar [SEED-Voken](https://github.com/TencentARC/SEED-Voken) y los tokenizadores Open-MAGVIT2 e IBQ
- [Hugging Face](https://huggingface.co) por alojar los modelos preentrenados
- Los equipos detrás de [VQGAN](https://github.com/CompVis/taming-transformers), [MAGVIT](https://github.com/google-research/magvit), [LlamaGen](https://github.com/FoundationVision/LlamaGen),[RQVAE](https://github.com/kakaobrain/rq-vae-transformer) y [VideoGPT](https://github.com/wilson1yan/VideoGPT), [OmniTokenizer](https://github.com/FoundationVision/OmniTokenizer).
