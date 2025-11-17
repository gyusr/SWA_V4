# Envroment
- OS: WSL1 Ubuntu 22.04.5 LTS
- GPU: Need
- python version: 3.10.x
- redis install need

# install
```bash
set venv && source ./.venv/bin/activate
pip install -r requirements.txt

# terminal 1
sudo service redis-server start
redis-cli ping

# terminal 2
celery -A celery_worker.celery_app worker -Q cpu_io --loglevel=info

# terminal 3
celery -A celery_worker.celery_app worker -Q gpu --pool=solo --loglevel=info --max-tasks-per-child=1

# terminal 4
uvicorn main:app --reload
```
