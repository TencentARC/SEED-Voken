from OpenImageTokenizer import MAGVIT2VideoTokenizer

def main():
    # Inicializar el tokenizador de video
    tokenizer_name = "TencentARC/Open-MAGVIT2-Tokenizer-262144-Video" 
    tokenizer = MAGVIT2VideoTokenizer(tokenizer_name)
    
    # Cargar el modelo
    model = tokenizer.load_model()
    print("Modelo cargado correctamente")
    
    # Directorio de salida
    output_dir = "resultados_video"
    
    # Procesar un video
    video_path = "/teamspace/studios/this_studio/Mono_en_una_sandia.mp4"
    results = tokenizer.process_video(video_path, output_dir, save_frames_interval=0.5, combine_segments=True)
    
    if results:
        print(f"Video procesado correctamente")
        print(f"Video original guardado en: {results['original']}")
        print(f"Video reconstruido guardado en: {results['reconstructed']}")
        print(f"Visualización de tokens guardada en: {results['tokens']}")
        print(f"Forma de los tokens: {results['token_shape']}")
    else:
        print("Error al procesar el video")

if __name__ == "__main__":
    main()