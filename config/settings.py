class Config:
    CAMERA_INDEX = 0
    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720
    FPS = 30

    MOTION_THRESHOLD = 25
    MIN_CONTOUR_AREA = 2500

    OLLAMA_MODEL = "llava"
    OLLAMA_HOST = "http://localhost:11434"
    
    VLM_PROMPT = (
        "Analiza el fotograma minuciosamente en espanol. "
        "Si observas personas, objetos en la mano o situaciones anómalas, "
        "responde en una frase corta. Si no, responde: 'Escena normal'."
    )
