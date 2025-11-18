import os
import subprocess
import time
import sys
import threading

def install_dependencies():
    print(">>> [SETUP] Installing System Dependencies...")
    subprocess.run(["apt-get", "update"], stdout=subprocess.DEVNULL)
    
    print(">>> [SETUP] Installing Python Libraries (this takes ~2 mins)...")
    pkgs = [
        "fastapi", "uvicorn", "python-multipart", "pyngrok", "nest_asyncio",
        "torch", "transformers", "accelerate", "bitsandbytes",
        "langchain", "langchain-community", "langchain-huggingface", 
        "langgraph", "requests", "termcolor"
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + pkgs)

def start_ngrok():
    from pyngrok import ngrok, conf
    
    # Load env vars manually since dotenv might not be installed yet
    token = "YOUR_NGROK_TOKEN_HERE" # REPLACE IF .ENV FAILS
    
    # Try reading from file
    if os.path.exists(".env"):
        with open(".env", "r") as f:
            for line in f:
                if line.startswith("NGROK_AUTH_TOKEN"):
                    token = line.split("=")[1].strip()

    if token == "your_ngrok_token_here":
        print("!!! WARNING: Ngrok Token not set in .env or script. Tunnel will fail.")
    
    conf.get_default().auth_token = token
    public_url = ngrok.connect(8000).public_url
    print(f"\n>>> [NETWORK] SYSTEM ONLINE. ACCESS DASHBOARD AT:\n>>> {public_url}/dashboard\n")
    return public_url

def run_server():
    print(">>> [CORE] Starting FastAPI Backend...")
    cmd = [sys.executable, "system.py"]
    subprocess.run(cmd)

def main():
    if not os.path.exists("system.py"):
        print("Error: system.py not found. Please generate the files first.")
        return

    install_dependencies()
    
    # Serve the dashboard HTML statically for simplicity via a small trick
    # We will append a route to system.py dynamically or just assume user runs it
    # For this 'Elite' setup, we assume the dashboard is opened locally pointing to the ngrok URL
    # OR we can serve it via FastAPI. Let's ensure system.py serves it.
    
    # Patch system.py to serve the HTML file if it doesn't
    with open("system.py", "r") as f:
        content = f.read()
    
    if "StaticFiles" not in content:
        patch = """
from fastapi.responses import FileResponse
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    with open("OKNLAB — Dashboard.html", "r") as f:
        return f.read()
"""
        with open("system.py", "a") as f:
            f.write(patch)

    # Start Ngrok
    public_url = start_ngrok()
    
    # Update Dashboard HTML with the dynamic API URL if needed (Client-side JS usually handles relative paths, 
    # but via Ngrok we treat it as relative to domain)
    
    # Start Server
    run_server()

if __name__ == "__main__":
    main()
