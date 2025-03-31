import os
from OpenImageTokenizer import IBQImageTokenizer

# Ruta a la imagen de prueba y directorio de salida
imagen_path = os.path.expanduser("~/testima.jpg")
output_dir = os.path.expanduser("~/resultados_tokens")

# Crear el tokenizador con el modelo deseado
print("Iniciando tokenizador...")
tokenizer = IBQImageTokenizer("TencentARC/IBQ-Tokenizer-1024")

# Cargar el modelo
tokenizer.load_model()

# Codificar la imagen
print(f"Procesando imagen: {imagen_path}")
encoded = tokenizer.encode(imagen_path)

# Obtener los tokens
tokens = encoded['indices']

# Visualizar los tokens (ahora con mejor manejo de formas 1D)
print("Visualizando tokens...")
tokenizer.visualize_tokens(tokens, save_path="tokens.png")

# También se puede procesar la imagen completa con un solo método
print("Procesando imagen completa...")
results = tokenizer.process_image(imagen_path, output_dir)

print("Procesamiento completado.")