FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y curl git build-essential
RUN curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y
ENV PATH="/root/.elan/bin:$PATH"
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python -c "
from huggingface_hub import snapshot_download
snapshot_download('YOUR_USERNAME/qed-ai-weights', local_dir='saved_model')
"
EXPOSE 7860
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "7860"]
