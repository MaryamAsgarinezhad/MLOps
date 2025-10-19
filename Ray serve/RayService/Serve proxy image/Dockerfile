FROM rayproject/ray:2.46.0

RUN pip install torch transformers
WORKDIR /app
COPY . /app/

ENV PYTHONPATH=/app:$PYTHONPATH

