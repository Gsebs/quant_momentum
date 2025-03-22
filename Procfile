web: cd src && gunicorn -w 1 -k uvicorn.workers.UvicornWorker wsgi:app --bind=0.0.0.0:$PORT --timeout 300
