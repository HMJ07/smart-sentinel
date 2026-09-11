class Config:
    CAMERA_INDEX = 0
    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720
    FPS = 30

    MOTION_THRESHOLD = 25
    MIN_CONTOUR_AREA = 2500

    OLLAMA_MODEL = "llama3.2-vision"
    OLLAMA_HOST = "http://localhost:11434"
    VLM_PROMPT = (
        "Analiza brevemente la escena. Responde en 1-2 frases en espanol: "
        "¿Hay alguna persona o evento sospechoso o inusual?"
    )
