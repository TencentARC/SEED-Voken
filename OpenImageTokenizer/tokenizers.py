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

class IBQImageTokenizer:
    """
    Tokenizador de imágenes basado en IBQ.
    Permite codificar imágenes en tokens y decodificar tokens de vuelta a imágenes.
    """
    def __init__(self, tokenizer, device=None):
        """
        Inicializa el tokenizador con un modelo específico.
        
        Args:
            tokenizer: Nombre del modelo (ej. "TencentARC/IBQ-Tokenizer-1024")
            device: Dispositivo para inferencia ("cuda", "cpu"). Si es None, se usa cuda si está disponible.
        """
        self.tokenizer = tokenizer
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.config = None
        self.checkpoint_path = None
        
        # Cargar la configuración
        self.config = get_model_config_IBQ(self.tokenizer)
        
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
        Carga el modelo IBQ usando la configuración y checkpoint.
    
        Returns:
            El modelo cargado
        """
        if self.model is not None:
            return self.model
    
        try:
            from OpenImageTokenizer.IBQ.models.ibqgan import IBQ
        
            # Obtener checkpoint y configuración
            checkpoint_path = self._get_checkpoint()
            config = self._get_config()
        
            print(f"Cargando modelo desde checkpoint: {checkpoint_path}")
        
            # Crear el modelo con los parámetros de configuración
            model_args = config["model"]["init_args"]
            self.model = IBQ(**model_args)
        
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
                    quant, qloss, (_, _, indices) = self.model.encode(img_tensor)
            else:
                quant, qloss, (_, _, indices) = self.model.encode(img_tensor)
    
            # Imprimir diagnóstico detallado
            print(f"Forma del tensor quant: {quant.shape}")
            print(f"Forma del tensor indices: {indices.shape if hasattr(indices, 'shape') else 'desconocida'}")

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

class MAGVIT2VideoTokenizer:
    """
    Tokenizador de videos basado en MAGVIT2.
    Permite codificar imágenes en tokens y decodificar tokens de vuelta a imágenes.
    """
    def __init__(self, tokenizer, device=None):
        self.tokenizer = tokenizer
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.config = None
        self.checkpoint_path = None
        
        # Cargar la configuración
        self.config = get_model_config_video(self.tokenizer)
        
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
        Carga el modelo MAGVIT2 para video usando la configuración y checkpoint.
    
        Returns:
            El modelo cargado
        """
        if self.model is not None:
            return self.model
    
        try:
            import os
            from OpenImageTokenizer.Open_MAGVIT2.models.video_lfqgan import VQModel
            from OpenImageTokenizer.hf_utils import get_model_checkpoint
        
            # Obtener checkpoint y configuración
            checkpoint_path = self._get_checkpoint()
            config = self._get_config()
        
            print(f"Cargando modelo desde checkpoint: {checkpoint_path}")
        
            # Crear el modelo con los parámetros de configuración ajustados
            model_args = config["model"]["init_args"].copy()
        
            # Verificar si se necesita un checkpoint de imagen preentrenado
            if "image_pretrain_path" in model_args:
                image_pretrain_path = model_args["image_pretrain_path"]
            
                # Si la ruta es relativa o no existe, obtener el checkpoint de imagen
                if not os.path.isabs(image_pretrain_path) or not os.path.exists(image_pretrain_path):
                    try:
                        # Intentar obtener el checkpoint de imagen correspondiente
                        image_model_name = "TencentARC/Open-MAGVIT2-Tokenizer-128-resolution"
                        print(f"Obteniendo checkpoint de imagen desde {image_model_name}...")
                        image_checkpoint_path = get_model_checkpoint(image_model_name)
                    
                        if os.path.exists(image_checkpoint_path):
                            model_args["image_pretrain_path"] = image_checkpoint_path
                            print(f"Usando checkpoint de imagen obtenido: {image_checkpoint_path}")
                        else:
                            print(f"No se pudo obtener el checkpoint de imagen. Desactivando inflate_from_image.")
                            model_args["image_pretrain_path"] = None
                    except Exception as e:
                        print(f"Error al obtener checkpoint de imagen: {e}")
                        model_args["image_pretrain_path"] = None
        
            # Crear el modelo con los argumentos ajustados
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

    def get_transforms(self, resolution=None):
        """
        Obtiene las transformaciones para preprocesar videos.

        Args:
            resolution: Resolución para redimensionar (si es None, se obtiene de la configuración)
        
        Returns:
            Transformaciones compuestas para videos
        """
        import OpenImageTokenizer.Open_MAGVIT2.data.video_transforms as video_transforms
        import OpenImageTokenizer.Open_MAGVIT2.data.volume_transforms as volume_transforms
    
        if resolution is None:
            config = self._get_config()
            if "resolution" in config["model"]["init_args"]:
                resolution = config["model"]["init_args"]["resolution"]
            else:
                resolution = 128  # Default fallback
    
        transforms = video_transforms.Compose([
            video_transforms.Resize(resolution, interpolation="bilinear"),
            video_transforms.CenterCrop(size=(resolution, resolution)),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # normalización a [-1, 1]
        ])
    
        return transforms

    def video_to_tensor(self, video_path, temporal_window=16, max_frames=64, resolution=None):
        """
        Convierte un video a un tensor normalizado para el modelo, limitando la cantidad de frames.
    
        Args:
            video_path: Ruta al video
            temporal_window: Número de frames por ventana temporal
            max_frames: Número máximo de frames a procesar para evitar problemas de memoria
            resolution: Resolución para redimensionar (si es None, se obtiene de la configuración)
            
        Returns:
            tuple: (tensor de video, frames por ventana, forma original)
        """
        from decord import VideoReader, cpu
        import numpy as np
    
        # Determinar resolución objetivo
        if resolution is None:
            config = self._get_config()
            if "resolution" in config["model"]["init_args"]:
                resolution = config["model"]["init_args"]["resolution"]
            else:
                resolution = 128  # Default fallback
    
        # Obtener las transformaciones
        transforms = self.get_transforms(resolution)
    
        # Cargar video
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
    
        # Limitar el número de frames para evitar problemas de memoria
        if total_frames > max_frames:
            print(f"Limitando el video de {total_frames} frames a {max_frames} frames para evitar problemas de memoria")
            # Tomar frames distribuidos uniformemente
            frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
        else:
            # Tomar todos los frames
            frame_indices = np.arange(0, total_frames, dtype=int)
    
        # Obtener frames seleccionados
        video_frames = vr.get_batch(frame_indices).asnumpy().astype(np.uint8)
    
        # Aplicar transformaciones
        video_tensor = transforms(video_frames)
    
        # Agregar dimensión de batch
        video_tensor = video_tensor.unsqueeze(0).to(self.device)
    
        return video_tensor, len(frame_indices), video_frames.shape

    def tensor_to_video(self, tensor):
        """
        Convierte un tensor de video a un array numpy para guardarlo.
    
        Args:
            tensor: Tensor de video normalizado [-1, 1]
            
        Returns:
            np.ndarray: Video en formato numpy [T, H, W, C]
        """
        # Constante para conversión a uint8
        _UINT8_MAX_F = float(torch.iinfo(torch.uint8).max)
    
        # Desacoplar del grafo y mover a CPU
        tensor = tensor.detach().cpu()
    
        # Limitar al rango [-1, 1]
        tensor = torch.clamp(tensor, -1., 1.)
    
        # Convertir a rango [0, 1]
        tensor = (tensor + 1.0) / 2.0
    
        # Reorganizar dimensiones [C, T, H, W] -> [T, H, W, C]
        video = tensor.permute(1, 2, 3, 0).numpy()
    
        # Escalar a [0, 255] y convertir a uint8
        video = (video * _UINT8_MAX_F + 0.5).astype(np.uint8)
    
        return video

    def save_video(self, video_array, output_path, fps=24):
        """
        Guarda un array de video en un archivo.
    
        Args:
            video_array: Array numpy del video [T, H, W, C]
            output_path: Ruta donde guardar el video
            fps: Frames por segundo
        """
        import mediapy as media
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        media.write_video(output_path, video_array, fps=fps)
        print(f"Video guardado en: {output_path}")

    def save_frames(self, video_path, output_dir, interval=0.5):
        """
        Extrae frames de un video y los guarda como imágenes.
    
        Args:
            video_path: Ruta al video
            output_dir: Directorio donde guardar los frames
            interval: Intervalo de tiempo (en segundos) entre frames a guardar
        """
        import cv2
        import imageio
    
        os.makedirs(output_dir, exist_ok=True)
    
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * interval)
    
        frame_count = 0
        saved_count = 0
    
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_path = os.path.join(output_dir, f"frame_{saved_count * interval:.1f}.png")
                imageio.imsave(frame_path, frame_rgb)
                saved_count += 1
            
            frame_count += 1
        
        cap.release()
        print(f"Se guardaron {saved_count} frames en {output_dir}")

    def encode(self, video, max_frames=64, chunk_size=16):
        """
        Codifica un video en tokens.

        Args:
            video: Ruta al video o tensor de video
            max_frames: Número máximo de frames a procesar a la vez
            chunk_size: Tamaño de los chunks temporales para procesar
    
        Returns:
            dict: {
                'quant': Representación cuantizada,
                'indices': Índices de tokens,
                'token_shape': Forma de los tokens
            }
        """
        # Cargar el modelo si aún no está cargado
        if self.model is not None or self.load_model() is not None:
            pass
        else:
            raise ValueError("No se pudo cargar el modelo")

        # Preparar el video según el tipo de entrada
        if isinstance(video, str):
            video_tensor, total_frames, original_shape = self.video_to_tensor(video, max_frames=max_frames)
        elif isinstance(video, torch.Tensor):
            if video.dim() == 4:  # [C, T, H, W]
                video_tensor = video.unsqueeze(0).to(self.device)  # Añadir dimensión de batch
            elif video.dim() == 5:  # [B, C, T, H, W]
                video_tensor = video.to(self.device)
            else:
                raise ValueError(f"Tensor de forma incorrecta: {video.shape}")
        else:
            raise ValueError("El parámetro video debe ser una ruta o un tensor")
    
        # Extraemos dimensiones
        b, c, t, h, w = video_tensor.shape
        print(f"Procesando video tensor de forma: {video_tensor.shape}")
    
        # Definimos el tamaño de la ventana temporal (según el modelo)
        temporal_window = min(chunk_size, t)  # Usar chunk_size o menos si el video es más corto
    
        # Lista para almacenar resultados
        quant_list = []
        indices_list = []
    
        try:
            # Procesar en trozos pequeños para evitar problemas de memoria
            # Codificar el video por ventanas temporales
            with torch.no_grad():
                for idx in range(0, (t - 1) // temporal_window + 1):
                    # Recortar ventana temporal
                    start = idx * temporal_window
                    end = min((idx + 1) * temporal_window, t)
                
                    print(f"Procesando segmento {idx+1}/{(t - 1) // temporal_window + 1} (frames {start}-{end-1})")
                
                    input_window = video_tensor[:, :, start:end, ...]
                
                    # Verificar si la ventana tiene frames suficientes
                    if input_window.shape[2] < 2:  # Necesitamos al menos 2 frames
                        print(f"Saltando segmento {idx+1} con solo {input_window.shape[2]} frames (necesitamos al menos 2)")
                        continue
                    
                    # Usar EMA si está disponible
                    if hasattr(self.model, 'use_ema') and self.model.use_ema:
                        with self.model.ema_scope():
                            quant, diff, indices, _ = self.model.encode(input_window)
                    else:
                        quant, diff, indices, _ = self.model.encode(input_window)
                
                    # Guardar resultados
                    quant_list.append(quant.cpu())  # Mover a CPU para liberar memoria
                    indices_list.append(indices.cpu())  # Mover a CPU para liberar memoria
                
                    # Forzar limpieza de memoria
                    torch.cuda.empty_cache()
        
            # Combinar resultados si hay múltiples ventanas
            if len(quant_list) > 1:
                # No intentar concatenar, mantener como lista para evitar problemas de memoria
                quant = quant_list
                indices = indices_list
            elif len(quant_list) == 1:
                quant = quant_list[0]
                indices = indices_list[0]
            else:
                raise ValueError("No se pudo codificar ninguna ventana temporal del video")
        
            # Obtener forma de los tokens
            token_shape = None
            if isinstance(indices, torch.Tensor):
                if indices.dim() == 5:  # [B, T, C, H, W]
                    token_shape = (indices.shape[1], indices.shape[3], indices.shape[4])  # [T, H, W]
                elif indices.dim() == 4:  # [B, T, H, W]
                    token_shape = (indices.shape[1], indices.shape[2], indices.shape[3])  # [T, H, W]
                elif indices.dim() == 3:  # [T, H, W]
                    token_shape = indices.shape
        
            # Imprimir información sobre la forma de los tokens
            print(f"Forma inferida de los tokens: {token_shape}")
        
            return {
                'quant': quant,
                'indices': indices,
                'token_shape': token_shape
            }
    
        except Exception as e:
            print(f"Error durante la codificación: {e}")
            # Liberar memoria en caso de error
            torch.cuda.empty_cache()
            raise

    def decode(self, quant, chunk_size=16):
        """
        Decodifica una representación cuantizada a un video.
    
        Args:
            quant: Representación cuantizada devuelta por encode
            chunk_size: Tamaño de los chunks temporales para procesar
        
        Returns:
            tensor: Tensor de video reconstruido
        """
        # Cargar el modelo si aún no está cargado
        if self.model is None:
            self.load_model()
    
        # Si tenemos una lista de quant (múltiples ventanas temporales)
        if isinstance(quant, list):
            reconstructed_list = []
        
            with torch.no_grad():
                for i, q in enumerate(quant):
                    print(f"Decodificando segmento {i+1}/{len(quant)}...")
                
                    # Mover al dispositivo correcto
                    q = q.to(self.device)
                
                    # Usar EMA si está disponible
                    if hasattr(self.model, 'use_ema') and self.model.use_ema:
                        with self.model.ema_scope():
                            reconstructed = self.model.decode(q)
                    else:
                        reconstructed = self.model.decode(q)
                
                    # Mover a CPU para liberar memoria
                    reconstructed_list.append(reconstructed.cpu())
                    torch.cuda.empty_cache()
            
            # No intentar concatenar para evitar problemas de memoria
            return reconstructed_list
            
        else:
            # Decodificar directamente
            with torch.no_grad():
                # Asegurar que está en el dispositivo correcto
                quant = quant.to(self.device)
            
                # Usar EMA si está disponible
                if hasattr(self.model, 'use_ema') and self.model.use_ema:
                    with self.model.ema_scope():
                        reconstructed = self.model.decode(quant)
                else:
                    reconstructed = self.model.decode(quant)
            
                return reconstructed

    def encode_decode(self, video, max_frames=64, chunk_size=16):
        """
        Codifica y decodifica un video (reconstrucción).
    
        Args:
            video: Ruta al video o tensor de video
            max_frames: Número máximo de frames a procesar a la vez
            chunk_size: Tamaño de los chunks temporales para procesar
        
        Returns:
            dict: {
                'original': Tensor de video original,
                'reconstructed': Tensor de video reconstruido,
                'indices': Índices de tokens,
                'token_shape': Forma de los tokens
            }
        """
        # Preparar el video según el tipo de entrada
        if isinstance(video, str):
            original_tensor, total_frames, original_shape = self.video_to_tensor(video, max_frames=max_frames)
        elif isinstance(video, torch.Tensor):
            original_tensor = video.to(self.device)
            if original_tensor.dim() == 4:  # [C, T, H, W]
                original_tensor = original_tensor.unsqueeze(0)  # [B, C, T, H, W]
        else:
            raise ValueError("El parámetro video debe ser una ruta o un tensor")
    
        # Codificar
        print("Codificando video...")
        encoded = self.encode(original_tensor, max_frames=max_frames, chunk_size=chunk_size)
    
        # Decodificar
        print("Decodificando video...")
        reconstructed = self.decode(encoded['quant'], chunk_size=chunk_size)
    
        return {
            'original': original_tensor,
            'reconstructed': reconstructed,
            'indices': encoded['indices'],
            'token_shape': encoded['token_shape']
        }

    def visualize_tokens_video(self, indices, save_path=None, token_size=16, colormap='viridis'):
        """
        Visualiza los tokens de video como imágenes para facilitar la interpretación.

        Args:
            indices: Índices de tokens (de encode)
            save_path: Directorio para guardar la visualización
            token_size: Tamaño de cada token en la visualización
            colormap: Mapa de colores a utilizar
    
        Returns:
            list: Lista de imágenes de visualización por frame
        """
        import matplotlib.pyplot as plt
        from PIL import Image
    
        # Si indices es una lista, procesamos cada elemento por separado
        if isinstance(indices, list):
            visualization_list = []
            for i, idx_tensor in enumerate(indices):
                print(f"Visualizando tokens del segmento {i+1}/{len(indices)}...")
                frame_path = None if save_path is None else os.path.join(save_path, f"segment_{i+1}")
            
                # Crear directorio si no existe
                if frame_path:
                    os.makedirs(frame_path, exist_ok=True)
            
                # Para cada tensor de índices en este segmento, intentar visualizarlo
                if isinstance(idx_tensor, torch.Tensor):
                    # Si es un tensor con dimensión temporal
                    if idx_tensor.dim() >= 3:  # [B, T, ...] o [T, ...]
                        # Determinar dimensión temporal
                        if idx_tensor.dim() >= 4:  # [B, T, ...]
                            temporal_dim = 1
                            num_frames = idx_tensor.shape[temporal_dim] if idx_tensor.shape[temporal_dim] > 1 else 1
                        else:  # [T, ...]
                            temporal_dim = 0
                            num_frames = idx_tensor.shape[temporal_dim] if idx_tensor.shape[temporal_dim] > 1 else 1
                    
                        # Visualizar cada frame
                        for t in range(num_frames):
                            if num_frames > 1:
                                if temporal_dim == 1:  # [B, T, ...]
                                    frame_indices = idx_tensor[:, t]
                                else:  # [T, ...]
                                    frame_indices = idx_tensor[t]
                            else:
                                # Si solo hay un frame, usar directamente
                                frame_indices = idx_tensor
                        
                            # Guardar visualización
                            frame_save_path = None if frame_path is None else os.path.join(frame_path, f"frame_{t:04d}")
                            try:
                                vis = self.visualize_tokens_frame(frame_indices, frame_save_path, token_size, colormap)
                                visualization_list.append(vis)
                            except Exception as e:
                                print(f"Error visualizando frame {t} del segmento {i+1}: {e}")
                    else:
                        # Sin dimensión temporal, visualizar directamente
                        try:
                            vis = self.visualize_tokens_frame(idx_tensor, frame_path, token_size, colormap)
                            visualization_list.append(vis)
                        except Exception as e:
                            print(f"Error visualizando segmento {i+1}: {e}")
            
            return visualization_list

    def visualize_tokens_frame(self, indices, save_path=None, token_size=16, colormap='viridis'):
        """
        Visualiza los tokens de un único frame como una imagen.

        Args:
            indices: Índices de tokens de un frame
            save_path: Ruta para guardar la visualización
            token_size: Tamaño de cada token en la visualización
            colormap: Mapa de colores a utilizar
    
        Returns:
            tuple: (visualización en escala de grises, visualización a color)
        """
        import matplotlib.pyplot as plt
        from PIL import Image
        import numpy as np
    
        # Convertir a numpy si es un tensor
        if isinstance(indices, torch.Tensor):
            indices = indices.detach().cpu()

        # Si indices tiene dimensión de batch, tomar el primer elemento
        if isinstance(indices, torch.Tensor) and indices.dim() > 2:
            indices = indices[0]
    
        # Intentar obtener una representación 2D
        if isinstance(indices, torch.Tensor):
            if indices.dim() == 1:
                # Intentar inferir una forma aproximadamente cuadrada
                total_tokens = indices.shape[0]
                size = int(np.sqrt(total_tokens))
                indices = indices[:size*size].reshape(size, size)
            elif indices.dim() > 2:
                # Si tiene dimensiones adicionales, aplanar a 2D
                h, w = indices.shape[-2], indices.shape[-1]
                indices = indices.reshape(-1, h, w)[-1]  # Tomar la última capa
    
        # Convertir a numpy
        if isinstance(indices, torch.Tensor):
            indices = indices.numpy()

        # Verificar que ahora tenemos una matriz 2D
        if not isinstance(indices, np.ndarray) or len(indices.shape) != 2:
            raise ValueError(f"Los índices deben ser una matriz 2D, pero tienen forma: {indices.shape}")

        # Obtener dimensiones
        h, w = indices.shape

        # Crear imagen base para visualización
        viz_img = np.zeros((h * token_size, w * token_size), dtype=np.uint8)

        # Obtener valor mínimo y máximo para normalización
        min_idx = np.min(indices)
        max_idx = np.max(indices)

        # Normalizar índices a rango [0, 255] para visualización
        if min_idx == max_idx:
            # Si todos los tokens son iguales, usar gris medio
            viz_img.fill(128)
        else:
            norm_indices = ((indices - min_idx) / (max_idx - min_idx) * 255).astype(np.uint8)
    
            # Llenar la visualización
            for i in range(h):
                for j in range(w):
                    viz_img[i*token_size:(i+1)*token_size, j*token_size:(j+1)*token_size] = norm_indices[i, j]

        # Crear versión a color
        color_img = None
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
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    Image.fromarray(viz_img).save(f"{save_path}_gray.png")
                    Image.fromarray(color_img[:,:,:3]).save(f"{save_path}_color.png")
        except Exception as e:
            print(f"Error al crear visualización a color: {e}")
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                Image.fromarray(viz_img).save(f"{save_path}_gray.png")
    
        return Image.fromarray(viz_img), Image.fromarray(color_img[:,:,:3]) if color_img is not None else None

    def combine_video_segments(self, segment_paths, output_path, fps=24):
        """
        Combina múltiples segmentos de video en un solo archivo.
    
        Args:
            segment_paths: Lista de rutas a los segmentos de video
            output_path: Ruta donde guardar el video combinado
            fps: Frames por segundo
    
        Returns:
            str: Ruta al video combinado
        """
        import cv2
        import numpy as np
        import mediapy as media
    
        # Verificar que hay segmentos para combinar
        if not segment_paths:
            print("No hay segmentos para combinar")
            return None
    
        # Lista para almacenar todos los frames
        all_frames = []
    
        # Procesar cada segmento
        for segment_path in segment_paths:
            # Abrir el video
            cap = cv2.VideoCapture(segment_path)
        
            # Verificar que se pudo abrir
            if not cap.isOpened():
                print(f"No se pudo abrir el segmento: {segment_path}")
                continue
        
            # Leer frames del segmento
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
            
                # Convertir de BGR a RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                all_frames.append(frame_rgb)
        
            # Cerrar el video
            cap.release()
    
        # Verificar que se obtuvieron frames
        if not all_frames:
            print("No se pudo extraer ningún frame de los segmentos")
            return None
    
        # Convertir a array numpy
        combined_video = np.array(all_frames)
    
        # Guardar video combinado
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        media.write_video(output_path, combined_video, fps=fps)
        print(f"Video combinado guardado en: {output_path}")
    
        return output_path

    def process_video(self, video_path, output_dir=None, save_frames_interval=None, max_frames=64, chunk_size=16, combine_segments=True):
        """
        Procesa un video: codifica, decodifica y visualiza tokens.
    
        Args:
            video_path: Ruta al video
            output_dir: Directorio para guardar resultados
            save_frames_interval: Intervalo para guardar frames (en segundos, None para no guardar)
            max_frames: Número máximo de frames a procesar
            chunk_size: Tamaño de los chunks temporales para procesar
            combine_segments: Si es True, combina los segmentos reconstruidos en un solo video
        
        Returns:
            dict: Información del procesamiento
        """
        # Crear directorios de salida si se especificó output_dir
        if output_dir:
            os.makedirs(os.path.join(output_dir, "original"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "reconstructed"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "tokens"), exist_ok=True)
    
        # Obtener nombre base del video
        video_name = os.path.basename(video_path)
        base_name = os.path.splitext(video_name)[0]
    
        try:
            # Codificar y decodificar el video
            print(f"Procesando video: {video_path}")
            results = self.encode_decode(video_path, max_frames=max_frames, chunk_size=chunk_size)
        
            # Guardar resultados si se especificó output_dir
            if output_dir:
                # Original - guardamos solo los primeros frames para evitar problemas de memoria
                original_video = self.tensor_to_video(results['original'][0])
                orig_path = os.path.join(output_dir, "original", f"{base_name}.mp4")
                self.save_video(original_video, orig_path)
                print(f"Video original guardado en: {orig_path}")
            
                # Reconstruido - puede ser una lista de tensores
                segment_paths = []
            
                if isinstance(results['reconstructed'], list):
                    # Guardar cada segmento como un archivo separado
                    for i, segment in enumerate(results['reconstructed']):
                        segment_video = self.tensor_to_video(segment[0])
                        segment_path = os.path.join(output_dir, "reconstructed", f"{base_name}_segment_{i+1}.mp4")
                        self.save_video(segment_video, segment_path)
                        print(f"Segmento reconstruido {i+1} guardado en: {segment_path}")
                        segment_paths.append(segment_path)
                
                    # La ruta principal apunta al primer segmento
                    rec_path = os.path.join(output_dir, "reconstructed", f"{base_name}_segment_1.mp4")
                
                    # Combinar segmentos si se solicita
                    if combine_segments and len(segment_paths) > 1:
                        combined_path = os.path.join(output_dir, "reconstructed", f"{base_name}_combined.mp4")
                        combined_path = self.combine_video_segments(segment_paths, combined_path)
                        if combined_path:
                            rec_path = combined_path  # Actualizar la ruta principal al video combinado
                            print(f"Video reconstruido completo: {rec_path}")
                else:
                    # Guardar como un solo archivo
                    reconstructed_video = self.tensor_to_video(results['reconstructed'][0])
                    rec_path = os.path.join(output_dir, "reconstructed", f"{base_name}.mp4")
                    self.save_video(reconstructed_video, rec_path)
                    print(f"Video reconstruido guardado en: {rec_path}")
            
                # Guardar frames si se especificó
                if save_frames_interval is not None:
                    orig_frames_dir = os.path.join(output_dir, "original", f"{base_name}_frames")
                    self.save_frames(orig_path, orig_frames_dir, interval=save_frames_interval)
                
                    if isinstance(results['reconstructed'], list) and not combine_segments:
                        # Guardar frames de cada segmento
                        for i in range(len(results['reconstructed'])):
                            segment_path = os.path.join(output_dir, "reconstructed", f"{base_name}_segment_{i+1}.mp4")
                            segment_frames_dir = os.path.join(output_dir, "reconstructed", f"{base_name}_segment_{i+1}_frames")
                            self.save_frames(segment_path, segment_frames_dir, interval=save_frames_interval)
                    else:
                        # Guardar frames del video combinado o único
                        rec_frames_dir = os.path.join(output_dir, "reconstructed", f"{base_name}_frames")
                        self.save_frames(rec_path, rec_frames_dir, interval=save_frames_interval)
            
                # Tokens - puede ser una lista de tensores
                if isinstance(results['indices'], list):
                    for i, indices_segment in enumerate(results['indices']):
                        token_dir = os.path.join(output_dir, "tokens", f"{base_name}_tokens_segment_{i+1}")
                        self.visualize_tokens_video(indices_segment, token_dir)
                
                    # La ruta principal apunta al directorio general de tokens
                    token_dir = os.path.join(output_dir, "tokens", f"{base_name}_tokens")
                else:
                    token_dir = os.path.join(output_dir, "tokens", f"{base_name}_tokens")
                    self.visualize_tokens_video(results['indices'], token_dir)
            
                return {
                    "original": orig_path,
                    "reconstructed": rec_path,
                    "tokens": token_dir,
                    "indices": results['indices'],
                    "token_shape": results['token_shape']
                }
            else:
                # Devolver resultados sin guardar archivos
                return {
                    "original": results['original'],
                    "reconstructed": results['reconstructed'],
                    "indices": results['indices'],
                    "token_shape": results['token_shape']
                }
                
        except Exception as e:
            print(f"Error procesando video {video_path}: {e}")
            import traceback
            traceback.print_exc()
            torch.cuda.empty_cache()  # Liberar memoria en caso de error
            return None