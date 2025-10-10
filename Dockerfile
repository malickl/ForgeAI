# =========================================
# 🐍 1️⃣ BASE : image Python officielle légère
# =========================================
FROM python:3.10-slim

# =========================================
# ⚙️ 2️⃣ Dossier de travail
# =========================================
WORKDIR /app

# =========================================
# 📦 3️⃣ Copie des fichiers nécessaires
# =========================================
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# =========================================
# 📦 4️⃣ Copie du reste du code
# =========================================
COPY . .

# =========================================
# 🚀 5️⃣ Commande de lancement FastAPI
# =========================================
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]