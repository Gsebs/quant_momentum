web: PYTHONPATH=$PYTHONPATH:$PWD/src gunicorn -w 1 -k uvicorn.workers.UvicornWorker src.app:app --bind=0.0.0.0:$PORT --timeout 300
