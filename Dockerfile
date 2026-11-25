# step 1 select python base image 
FROM python:3.13-slim

# step 2 set working directory
WORKDIR /app

# step 3 copy requirement file
COPY requirements.txt .

# step 4 install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# step 5 copy backend code
COPY backend/ ./backend/

# step 6 copy trained model
COPY ml/model.pkl ./ml/model.pkl

# step 7 expose API port
EXPOSE 8000

# step 8 start FastAPI server
# CMD ["python","-m","uvicorn","backend.main:app","--host","0.0.0.0","--port","8000"]

CMD ["sh", "-c", "python -m uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]