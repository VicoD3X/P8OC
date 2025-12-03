import os
import subprocess

def main():
    # Par défaut : on lance Streamlit (app front)
    mode = os.getenv("APP_MODE", "streamlit")
    port = os.getenv("PORT", "8000")

    if mode == "api":
        # Lancer l'API FastAPI
        subprocess.run([
            "uvicorn", "main:app",
            "--host", "0.0.0.0",
            "--port", port,
        ])
    else:
        # Lancer l'app Streamlit
        subprocess.run([
            "streamlit", "run", "app/streamlit_app.py",
            "--server.port", port,
            "--server.address", "0.0.0.0",
        ])

if __name__ == "__main__":
    main()
