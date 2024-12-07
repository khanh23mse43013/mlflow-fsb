FROM tiangolo/uvicorn-gunicorn-fastapi

ENV DD_TRACE_ENABLED 0

WORKDIR /app/services/mlops-service

#install makerfile
RUN apt-get update && apt-get install -y make

COPY requirements.txt .

COPY . .

RUN make my_env_init

RUN make install
