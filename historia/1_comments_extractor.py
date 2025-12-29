# main.py (en la raíz del proyecto)
from dotenv import load_dotenv
load_dotenv()  # carga variables del archivo .env al entorno del proceso

import os
from services.openai import ask_openai

# DEBUG: mostrar exactamente qué está llegando (usa repr para ver espacios/escapes)
print("OPENAI_API_KEY repr:", repr(os.getenv("OPENAI_API_KEY")))

def main():
    pregunta = "¿Cuál es la capital de Francia?"
    try:
        respuesta = ask_openai(pregunta, temperature=0.0, model="gpt-4.1")
        print("Respuesta raw del servicio:\n", respuesta)
    except RuntimeError as e:
        print("Error al llamar al servicio OpenAI:", e)

if __name__ == "__main__":
    main()
