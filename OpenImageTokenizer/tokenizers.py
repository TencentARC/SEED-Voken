from .IBQ import *
from .Open_MAGVIT2 import *
from .hf_utils import *
from .configs import *
import torch
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt


class MAGVIT2ImageTokenizer:
    """
    Tokenizador de imágenes basado en MAGVIT2.
    Permite codificar imágenes en tokens y decodificar tokens de vuelta a imágenes.
    """
    
    def __init__(self, tokenizer, device=None):
        """
        Inicializa el tokenizador con un modelo específico.
        
        Args:
            tokenizer: Nombre del modelo (ej. "TencentARC/Open-MAGVIT2-Tokenizer-256-resolution")
            device: Dispositivo para inferencia ("cuda", "cpu"). Si es None, se usa cuda si está disponible.
        """
        self.tokenizer = tokenizer
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.config = None
        self.checkpoint_path = None
        
        # Cargar la configuración
        self.config = get_model_config(self.tokenizer)
        
        # Imprimir información básica
        print(f"Tokenizador inicializado para el modelo: {tokenizer}")
        print(f"Usando dispositivo: {self.device}")
    
    def _get_checkpoint(self):
        """Obtiene la ruta al archivo de checkpoint del modelo"""
        if self.checkpoint_path is None:
            self.checkpoint_path = get_model_checkpoint(self.tokenizer)
        return self.checkpoint_path

    def _get_config(self):
        """Obtiene la configuración del modelo"""
        return self.config
    
    def load_model(self):
        """
        Carga el modelo MAGVIT2 usando la configuración y checkpoint.
        
        Returns:
            El modelo cargado
        """
        if self.model is not None:
            return self.model
        
        try:
            
            from OpenImageTokenizer.Open_MAGVIT2.models.lfqgan import VQModel
            
            # Obtener checkpoint y configuración
            checkpoint_path = self._get_checkpoint()
            config = self._get_config()
            
            print(f"Cargando modelo desde checkpoint: {checkpoint_path}")
            
            # Crear el modelo con los parámetros de configuración
            model_args = config["model"]["init_args"]
            self.model = VQModel(**model_args)
            
            # Cargar pesos del checkpoint
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            
            # Cargar pesos en el modelo
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            
            if len(missing) > 0:
                print(f"Claves faltantes (probablemente sólo de loss y discriminator): {len(missing)} claves")
            if len(unexpected) > 0:
                print(f"Claves inesperadas: {len(unexpected)} claves")
            
            # Mover modelo al dispositivo y poner en modo evaluación
            self.model = self.model.eval().to(self.device)
            print("Modelo cargado correctamente")
            
            return self.model
            
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def image_to_tensor(self, image_path, target_size=None):
        """
        Convierte una imagen a un tensor normalizado para el modelo.
        
        Args:
            image_path: Ruta a la imagen o objeto PIL Image
            target_size: Tamaño objetivo para redimensionar (si es None, se obtiene de la configuración)
            
        Returns:
            tuple: (tensor de imagen, imagen PIL original)
        """
        # Determinar tamaño objetivo
        if target_size is None:
            config = self._get_config()
            if "resolution" in config["model"]["init_args"]:
                target_size = config["model"]["init_args"]["resolution"]
            else:
                target_size = 256  # Default fallback
        
        # Cargar imagen
        if isinstance(image_path, str):
            img = Image.open(image_path).convert('RGB')
        elif isinstance(image_path, Image.Image):
            img = image_path.convert('RGB')
        else:
            raise ValueError("image_path debe ser una ruta a un archivo o una imagen PIL")
        
        # Redimensionar si es necesario
        if img.size != (target_size, target_size):
            print(f"Redimensionando imagen de {img.size} a ({target_size}, {target_size})")
            img = img.resize((target_size, target_size), Image.LANCZOS)
        
        # Convertir a tensor y normalizar a [-1, 1]
        img_tensor = torch.from_numpy(np.array(img)).float()
        img_tensor = img_tensor / 127.5 - 1.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # [B, C, H, W]
        
        return img_tensor.to(self.device), img
    
    def tensor_to_image(self, tensor):
        """
        Convierte un tensor de imagen a una imagen PIL.
        
        Args:
            tensor: Tensor de imagen normalizado [-1, 1]
            
        Returns:
            PIL.Image: Imagen convertida
        """
        # Desacoplar del grafo y mover a CPU
        x = tensor.detach().cpu()
        
        # Limitar al rango [-1, 1]
        x = torch.clamp(x, -1., 1.)
        
        # Convertir a rango [0, 1]
        x = (x + 1.)/2.
        
        # Reorganizar dimensiones [C, H, W] -> [H, W, C]
        x = x.permute(1, 2, 0).numpy()
        
        # Escalar a [0, 255] y convertir a uint8
        x = (255*x).astype(np.uint8)
        
        # Crear imagen PIL
        img = Image.fromarray(x)
        if img.mode != "RGB":
            img = img.convert("RGB")
            
        return img
    
    def encode(self, image):
        """
        Codifica una imagen en tokens.
    
        Args:
            image: Ruta a la imagen, imagen PIL, o tensor de imagen
        
        Returns:
            dict: {
                'quant': Representación cuantizada,
                'indices': Índices de tokens,
                'token_shape': Forma de los tokens
            }
        """
        # Cargar el modelo si aún no está cargado
        if self.model is None:
            self.load_model()
    
        # Preparar la imagen según el tipo de entrada
        if isinstance(image, str) or isinstance(image, Image.Image):
            img_tensor, original_img = self.image_to_tensor(image)
        elif isinstance(image, torch.Tensor):
            if image.dim() == 3:  # [C, H, W]
                img_tensor = image.unsqueeze(0).to(self.device)  # Añadir dimensión de batch
            elif image.dim() == 4:  # [B, C, H, W]
                img_tensor = image.to(self.device)
            else:
                raise ValueError(f"Tensor de forma incorrecta: {image.shape}")
        else:
            raise ValueError("El parámetro image debe ser una ruta, una imagen PIL o un tensor")
    
        # Codificar la imagen
        with torch.no_grad():
            # Usar EMA si está disponible
            if hasattr(self.model, 'use_ema') and self.model.use_ema:
                with self.model.ema_scope():
                    encode_result = self.model.encode(img_tensor)
            else:
                encode_result = self.model.encode(img_tensor)
        
            # Imprimir diagnóstico detallado
            print(f"Tipo de resultado de encodificación: {type(encode_result)}")
            if isinstance(encode_result, tuple):
                print(f"Longitud del resultado: {len(encode_result)}")
                for i, item in enumerate(encode_result):
                    print(f"  Elemento {i}: tipo {type(item)}, forma {item.shape if hasattr(item, 'shape') else 'desconocida'}")
        
            # Extraer resultados (manejar diferentes formatos de retorno)
            if isinstance(encode_result, tuple):
                if len(encode_result) >= 3:
                    quant = encode_result[0]
                    indices = encode_result[2]  # Los índices suelen estar en la posición 2
                    print(f"Forma del tensor quant: {quant.shape}")
                    print(f"Forma del tensor indices: {indices.shape}")
                else:
                    raise ValueError(f"Formato de retorno de encode inesperado: tupla con {len(encode_result)} elementos, se esperaban al menos 3")
            else:
                raise ValueError(f"Formato de retorno de encode inesperado: {type(encode_result)}")
    
        # Obtener forma de los tokens
        if isinstance(indices, torch.Tensor):
            if indices.dim() == 4:  # [B, C, H, W]
                token_shape = indices.shape[-2:]  # [H, W]
            elif indices.dim() == 3:  # [B, H, W]
                token_shape = indices.shape[-2:]  # [H, W]
            elif indices.dim() == 2:  # [H, W]
                token_shape = indices.shape
            elif indices.dim() == 1:  # [N]
                # Intentar convertir a forma cuadrada
                size = int(np.sqrt(indices.shape[0]))
                token_shape = (size, size)
            else:
                token_shape = None
        else:
            token_shape = None
    
        # Imprimir información sobre la forma de los tokens
        print(f"Forma inferida de los tokens: {token_shape}")
    
        return {
            'quant': quant,
            'indices': indices,
            'token_shape': token_shape
        }
    
    def decode(self, quant):
        """
        Decodifica una representación cuantizada a una imagen.
        
        Args:
            quant: Representación cuantizada devuelta por encode
            
        Returns:
            tensor: Tensor de imagen reconstruida
        """
        # Cargar el modelo si aún no está cargado
        if self.model is None:
            self.load_model()
        
        # Decodificar
        with torch.no_grad():
            # Usar EMA si está disponible
            if hasattr(self.model, 'use_ema') and self.model.use_ema:
                with self.model.ema_scope():
                    decoded = self.model.decode(quant)
            else:
                decoded = self.model.decode(quant)
        
        return decoded
    
    def encode_decode(self, image):
        """
        Codifica y decodifica una imagen (reconstrucción).
        
        Args:
            image: Ruta a la imagen, imagen PIL, o tensor de imagen
            
        Returns:
            dict: {
                'original': Tensor de imagen original,
                'reconstructed': Tensor de imagen reconstruida,
                'indices': Índices de tokens,
                'token_shape': Forma de los tokens
            }
        """
        # Preparar la imagen según el tipo de entrada
        if isinstance(image, str) or isinstance(image, Image.Image):
            original_tensor, original_img = self.image_to_tensor(image)
        elif isinstance(image, torch.Tensor):
            original_tensor = image.to(self.device)
            if original_tensor.dim() == 3:  # [C, H, W]
                original_tensor = original_tensor.unsqueeze(0)  # [B, C, H, W]
        else:
            raise ValueError("El parámetro image debe ser una ruta, una imagen PIL o un tensor")
        
        # Codificar
        encoded = self.encode(original_tensor)
        
        # Decodificar
        reconstructed = self.decode(encoded['quant'])
        
        return {
            'original': original_tensor,
            'reconstructed': reconstructed,
            'indices': encoded['indices'],
            'token_shape': encoded['token_shape']
        }
    
    def visualize_tokens(self, indices, save_path=None, token_size=16, colormap='viridis'):
        """
        Visualiza los tokens como una imagen para facilitar la interpretación.
    
        Args:
            indices: Índices de tokens (de encode)
            save_path: Ruta para guardar la visualización
            token_size: Tamaño de cada token en la visualización
            colormap: Mapa de colores a utilizar
        
        Returns:
            tuple: (visualización en escala de grises, visualización a color)
        """
        # Convertir a numpy si es un tensor
        if isinstance(indices, torch.Tensor):
            indices = indices.detach().cpu()
    
        # Imprimir diagnóstico
        print(f"Forma original de los tokens: {indices.shape if hasattr(indices, 'shape') else 'desconocido'}")
        print(f"Tipo de tokens: {type(indices)}")
    
        # Si indices tiene dimensión de batch, tomar el primer elemento
        if isinstance(indices, torch.Tensor) and indices.dim() > 2:
            indices = indices[0]
            print(f"Usando primer elemento del batch: {indices.shape}")
    
        # Manejar tensores 1D convirtiéndolos a 2D
        if isinstance(indices, torch.Tensor) and indices.dim() == 1:
            # Intentar inferir una forma aproximadamente cuadrada
            total_tokens = indices.shape[0]
            size = int(np.sqrt(total_tokens))
            # Asegurarse de que size*size <= total_tokens
            indices = indices[:size*size]  # Recortar si es necesario
            indices = indices.reshape(size, size)
            print(f"Tensor 1D convertido a forma 2D: {indices.shape}")
    
        # Convertir a numpy
        if isinstance(indices, torch.Tensor):
            indices = indices.numpy()
    
        # Verificar que ahora tenemos una matriz 2D
        if not isinstance(indices, np.ndarray) or len(indices.shape) != 2:
            raise ValueError(f"Los índices deben ser una matriz 2D, pero tienen forma: {indices.shape if hasattr(indices, 'shape') else 'desconocida'}")
    
        # Obtener dimensiones
        h, w = indices.shape
        print(f"Forma final de tokens para visualización: {h}x{w}")
    
        # Crear imagen base para visualización
        viz_img = np.zeros((h * token_size, w * token_size), dtype=np.uint8)
    
        # Obtener valor mínimo y máximo para normalización
        min_idx = np.min(indices)
        max_idx = np.max(indices)
    
        # Normalizar índices a rango [0, 255] para visualización
        if min_idx == max_idx:
            # Si todos los tokens son iguales, usar gris medio
            viz_img.fill(128)
            print(f"Todos los tokens tienen el mismo valor: {min_idx}")
        else:
            norm_indices = ((indices - min_idx) / (max_idx - min_idx) * 255).astype(np.uint8)
        
            # Llenar la visualización
            for i in range(h):
                for j in range(w):
                    viz_img[i*token_size:(i+1)*token_size, j*token_size:(j+1)*token_size] = norm_indices[i, j]
    
        # Crear versión a color
        try:
            if min_idx != max_idx:
                norm_float = (indices - min_idx) / (max_idx - min_idx)
                colored = plt.cm.get_cmap(colormap)(norm_float)
                colored = (colored * 255).astype(np.uint8)
            
                color_img = np.zeros((h * token_size, w * token_size, 4), dtype=np.uint8)
                for i in range(h):
                    for j in range(w):
                        color_img[i*token_size:(i+1)*token_size, j*token_size:(j+1)*token_size] = colored[i, j]
            
                # Si se especificó ruta, guardar ambas imágenes
                if save_path:
                    Image.fromarray(viz_img).save(save_path)
                    color_path = save_path.replace('.png', '_color.png')
                    Image.fromarray(color_img[:,:,:3]).save(color_path)
                    print(f"Visualización guardada en: {save_path} y {color_path}")
            
                # Mostrar estadísticas
                unique_tokens = len(np.unique(indices))
                total_tokens = indices.size
                print(f"Tokens únicos: {unique_tokens}/{total_tokens} ({unique_tokens/total_tokens*100:.2f}%)")
                print(f"Rango de tokens: {min_idx} - {max_idx}")
            
                # Devolver ambas imágenes
                return Image.fromarray(viz_img), Image.fromarray(color_img[:,:,:3])
            else:
                # Si todos los tokens son iguales
                if save_path:
                    Image.fromarray(viz_img).save(save_path)
                    print(f"Visualización guardada en: {save_path}")
                return Image.fromarray(viz_img), None
        except Exception as e:
            print(f"Error al crear visualización a color: {e}")
            import traceback
            traceback.print_exc()
            if save_path:
                Image.fromarray(viz_img).save(save_path)
            return Image.fromarray(viz_img), None
    
    def process_image(self, image_path, output_dir=None):
        """
        Procesa una imagen: codifica, decodifica y visualiza tokens.
        
        Args:
            image_path: Ruta a la imagen
            output_dir: Directorio para guardar resultados
            
        Returns:
            dict: Información del procesamiento
        """
        # Crear directorios de salida si se especificó output_dir
        if output_dir:
            os.makedirs(os.path.join(output_dir, "original"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "reconstructed"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "tokens"), exist_ok=True)
        
        # Obtener nombre base de la imagen
        img_name = os.path.basename(image_path)
        base_name = os.path.splitext(img_name)[0]
        
        try:
            # Codificar y decodificar la imagen
            results = self.encode_decode(image_path)
            
            # Guardar resultados si se especificó output_dir
            if output_dir:
                # Original
                original_img = self.tensor_to_image(results['original'][0])
                orig_path = os.path.join(output_dir, "original", f"{base_name}.png")
                original_img.save(orig_path)
                print(f"Imagen original guardada en: {orig_path}")
                
                # Reconstruida
                reconstructed_img = self.tensor_to_image(results['reconstructed'][0])
                rec_path = os.path.join(output_dir, "reconstructed", f"{base_name}.png")
                reconstructed_img.save(rec_path)
                print(f"Imagen reconstruida guardada en: {rec_path}")
                
                # Tokens
                token_path = os.path.join(output_dir, "tokens", f"{base_name}_tokens.png")
                self.visualize_tokens(results['indices'], token_path)
                
                return {
                    "original": orig_path,
                    "reconstructed": rec_path,
                    "tokens": token_path,
                    "indices": results['indices'],
                    "token_shape": results['token_shape']
                }
            else:
                # Devolver resultados sin guardar archivos
                return {
                    "original": self.tensor_to_image(results['original'][0]),
                    "reconstructed": self.tensor_to_image(results['reconstructed'][0]),
                    "indices": results['indices'],
                    "token_shape": results['token_shape']
                }
                
        except Exception as e:
            print(f"Error procesando imagen {image_path}: {e}")
            import traceback
            traceback.print_exc()
            return None