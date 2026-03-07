FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY langraph_rag_backend.py .
COPY langraph_rag_frontend.py .
COPY .env .
EXPOSE 8501
CMD ["streamlit", "run", "langraph_rag_frontend.py", "--server.port=8501", "--server.address=0.0.0.0"]