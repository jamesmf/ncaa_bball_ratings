FROM python:3.10

WORKDIR /app/ratings/

RUN python -m pip install uv


ENV VIRTUAL_ENV=/usr/local/

COPY ./pyproject.toml .

COPY . .
RUN uv pip install -e .