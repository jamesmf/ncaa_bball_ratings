# FROM nvidia/cuda:11.7.0-runtime-ubuntu20.04
FROM python:3.10

WORKDIR /app/ratings/

# RUN ln -s /usr/bin/python3.10 /usr/bin/python \ 
    # && curl -LsSf https://astral.sh/uv/install.sh > install.sh \
    # && chmod -R 655 install.sh \
    # && ./install.sh 
RUN python -m pip install uv

COPY ./pyproject.toml .
# RUN 

COPY . .
RUN uv pip install --system -e .