web: gunicorn -w 1 -k uvicorn.workers.UvicornWorker fastapi_app:app --bind=0.0.0.0:$PORT --timeout 300 --log-level debug
