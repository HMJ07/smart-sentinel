import asyncio
import cv2
import ollama
from config.settings import Config

class VLMAnalyzer:
    def __init__(self):
        self.is_analyzing = False
        self.latest_analysis = "Esperando deteccion de movimiento..."

    async def analyze_frame_async(self, frame):
        if self.is_analyzing:
            return

        self.is_analyzing = True
        try:
            _, buffer = cv2.imencode('.jpg', frame)
            image_bytes = buffer.tobytes()

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: ollama.generate(
                    model=Config.OLLAMA_MODEL,
                    prompt=Config.VLM_PROMPT,
                    images=[image_bytes]
                )
            )
            self.latest_analysis = response.get('response', 'Sin respuesta del VLM.').strip()
        except Exception as e:
            self.latest_analysis = f"Error VLM: {str(e)}"
        finally:
            self.is_analyzing = False
