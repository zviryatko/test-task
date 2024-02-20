FROM python:3.11-slim-buster
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache \
    pip install -Ur requirements.txt
ADD . /app
WORKDIR /app
ENV PATH="${PATH}:/usr/local/lib/python3.11/site-packages"
EXPOSE 5000
CMD ["flask", "run", "--reload", "--host", "0.0.0.0", "--port", "5000"]
