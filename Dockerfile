FROM python:3.10

WORKDIR /app/ratings/

RUN python -m pip install uv

COPY ./pyproject.toml .

COPY . .
RUN uv pip install --system -e .