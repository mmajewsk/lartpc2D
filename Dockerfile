FROM anibali/pytorch:1.5.0-cuda10.2
RUN python3 -m pip install gym gym-lartpc mlflow neptune-client
COPY . /app
