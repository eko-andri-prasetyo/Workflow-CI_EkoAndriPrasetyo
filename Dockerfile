FROM python:3.12-slim

WORKDIR /app

# copy hanya dependency dulu biar cache build enak
COPY MLProject/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# copy project
COPY MLProject /app/MLProject

# default command: training (modelling.py)
CMD ["python", "MLProject/modelling.py", "--data-path", "MLProject/creditscoring_preprocessing/creditscoring_preprocessed.csv", "--target-col", "target"]
