"""
Utilidades para descargar modelos de Hugging Face
"""
import os
import logging
import shutil
from pathlib import Path
from urllib.parse import urlparse
import requests
from tqdm import tqdm

# Configurar logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("hf_utils")

# Mapeo de modelos a archivos checkpoint
MODEL_CHECKPOINT_MAPPING = {
    # Open-MAGVIT2 models
    "TencentARC/Open-MAGVIT2-Tokenizer-128-resolution": "imagenet_128_L.ckpt",
    "TencentARC/Open-MAGVIT2-Tokenizer-256-resolution": "imagenet_256_L.ckpt",
    "TencentARC/Open-MAGVIT2-Tokenizer-16384-Pretrain": "pretrain256_16384.ckpt",
    "TencentARC/Open-MAGVIT2-Tokenizer-262144-Pretrain": "pretrain256_262144.ckpt",
    
    # IBQ models
    "TencentARC/IBQ-Tokenizer-1024": "imagenet256_1024.ckpt",
    "TencentARC/IBQ-Tokenizer-8192": "imagenet256_8192.ckpt",
    "TencentARC/IBQ-Tokenizer-16384": "imagenet256_16384.ckpt",
    "TencentARC/IBQ-Tokenizer-262144": "imagenet256_262144.ckpt"
}

# URLs directas para descargar los checkpoints
MODEL_URLS = {
    "TencentARC/Open-MAGVIT2-Tokenizer-128-resolution": "https://huggingface.co/TencentARC/Open-MAGVIT2-Tokenizer-128-resolution/resolve/main/imagenet_128_L.ckpt",
    "TencentARC/Open-MAGVIT2-Tokenizer-256-resolution": "https://huggingface.co/TencentARC/Open-MAGVIT2-Tokenizer-256-resolution/resolve/main/imagenet_256_L.ckpt",
    "TencentARC/Open-MAGVIT2-Tokenizer-16384-Pretrain": "https://huggingface.co/TencentARC/Open-MAGVIT2-Tokenizer-16384-Pretrain/resolve/main/pretrain256_16384.ckpt",
    "TencentARC/Open-MAGVIT2-Tokenizer-262144-Pretrain": "https://huggingface.co/TencentARC/Open-MAGVIT2-Tokenizer-262144-Pretrain/resolve/main/pretrain256_262144.ckpt",
    "TencentARC/IBQ-Tokenizer-1024": "https://huggingface.co/TencentARC/IBQ-Tokenizer-1024/resolve/main/imagenet256_1024.ckpt",
    "TencentARC/IBQ-Tokenizer-8192": "https://huggingface.co/TencentARC/IBQ-Tokenizer-8192/resolve/main/imagenet256_8192.ckpt",
    "TencentARC/IBQ-Tokenizer-16384": "https://huggingface.co/TencentARC/IBQ-Tokenizer-16384/resolve/main/imagenet256_16384.ckpt",
    "TencentARC/IBQ-Tokenizer-262144": "https://huggingface.co/TencentARC/IBQ-Tokenizer-262144/resolve/main/imagenet256_262144.ckpt"
}

def get_default_cache_dir():
    """
    Obtiene el directorio de caché predeterminado para almacenar los modelos descargados
    """
    # Usar el directorio del usuario por defecto
    home_dir = os.path.expanduser("~")
    cache_dir = os.path.join(home_dir, ".cache", "image_tokenizers")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

def download_file(url, destination):
    """
    Descarga un archivo de una URL a un destino con barra de progreso
    """
    # Crear el directorio de destino si no existe
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    # Descargar con barra de progreso
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    
    logger.info(f"Descargando {url} a {destination}")
    temp_file = destination + ".download"
    
    try:
        with open(temp_file, 'wb') as f, tqdm(
                total=total_size, unit='iB', unit_scale=True,
                desc=os.path.basename(destination)) as bar:
            for data in response.iter_content(block_size):
                size = f.write(data)
                bar.update(size)
        
        # Renombrar el archivo temporal al nombre final
        shutil.move(temp_file, destination)
        return True
    except Exception as e:
        logger.error(f"Error descargando archivo: {e}")
        if os.path.exists(temp_file):
            os.remove(temp_file)
        return False

def get_checkpoint_path(model_name, cache_dir=None):
    """
    Obtiene la ruta al archivo checkpoint para un modelo dado.
    Si el checkpoint no existe localmente, lo descarga desde Hugging Face.
    
    Args:
        model_name: Nombre del modelo en formato "organización/nombre"
        cache_dir: Directorio de caché para almacenar los checkpoints descargados
        
    Returns:
        str: Ruta al archivo checkpoint
    """
    if model_name not in MODEL_CHECKPOINT_MAPPING:
        raise ValueError(f"Modelo no soportado: {model_name}. Los modelos soportados son: {list(MODEL_CHECKPOINT_MAPPING.keys())}")
    
    # Usar directorio de caché predeterminado si no se especifica
    if cache_dir is None:
        cache_dir = get_default_cache_dir()
    
    # Obtener el nombre del archivo checkpoint
    checkpoint_filename = MODEL_CHECKPOINT_MAPPING[model_name]
    
    # Construir la ruta completa
    checkpoint_dir = os.path.join(cache_dir, model_name.replace("/", "_"))
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
    
    # Verificar si el archivo ya existe
    if os.path.exists(checkpoint_path):
        logger.info(f"El checkpoint ya existe en {checkpoint_path}")
        return checkpoint_path
    
    # Si no existe, descargarlo
    logger.info(f"Checkpoint no encontrado. Descargando {model_name}...")
    
    # Crear el directorio si no existe
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Obtener la URL de descarga
    if model_name in MODEL_URLS:
        url = MODEL_URLS[model_name]
    else:
        # Construir la URL basada en la convención de Hugging Face
        url = f"https://huggingface.co/{model_name}/resolve/main/{checkpoint_filename}"
    
    # Descargar el archivo
    success = download_file(url, checkpoint_path)
    
    if success:
        logger.info(f"Checkpoint descargado exitosamente en {checkpoint_path}")
        return checkpoint_path
    else:
        raise RuntimeError(f"Error al descargar el checkpoint para {model_name}")

class HFModelManager:
    """
    Administrador para modelos de Hugging Face.
    Proporciona métodos para descargar y acceder a los checkpoints de los modelos.
    """
    
    def __init__(self, cache_dir=None):
        """
        Inicializa el administrador de modelos.
        
        Args:
            cache_dir: Directorio de caché para almacenar los modelos descargados
        """
        self.cache_dir = cache_dir or get_default_cache_dir()
        self.downloaded_models = {}
    
    def get_model_path(self, model_name):
        """
        Obtiene la ruta al checkpoint del modelo.
        
        Args:
            model_name: Nombre del modelo en formato "organización/nombre"
            
        Returns:
            str: Ruta al archivo checkpoint
        """
        # Verificar si ya tenemos la ruta en caché
        if model_name in self.downloaded_models:
            return self.downloaded_models[model_name]
        
        # Obtener la ruta del checkpoint (descargando si es necesario)
        checkpoint_path = get_checkpoint_path(model_name, self.cache_dir)
        
        # Guardar la ruta en caché
        self.downloaded_models[model_name] = checkpoint_path
        
        return checkpoint_path
    
    def list_available_models(self):
        """
        Lista todos los modelos disponibles para descarga
        
        Returns:
            list: Lista de nombres de modelos disponibles
        """
        return list(MODEL_CHECKPOINT_MAPPING.keys())
    
    def list_downloaded_models(self):
        """
        Lista todos los modelos ya descargados
        
        Returns:
            dict: Diccionario con nombres de modelos como claves y rutas como valores
        """
        return self.downloaded_models.copy()

# Instancia global del administrador de modelos
model_manager = HFModelManager()

# Función para facilitar el acceso al administrador global
def get_model_checkpoint(model_name):
    """
    Obtiene la ruta al checkpoint de un modelo dado.
    Esta es una función de conveniencia que utiliza el administrador global.
    
    Args:
        model_name: Nombre del modelo en formato "organización/nombre"
        
    Returns:
        str: Ruta al archivo checkpoint
    """
    return model_manager.get_model_path(model_name)
