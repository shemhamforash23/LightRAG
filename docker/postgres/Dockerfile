FROM postgres:15

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        postgresql-server-dev-15 \
        git \
        ca-certificates \
        cmake \
        libssl-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

RUN cd /tmp \
    && git clone --branch v0.5.1 https://github.com/pgvector/pgvector.git \
    && cd pgvector \
    && make \
    && make install \
    && cd .. \
    && rm -rf pgvector

COPY init.sql /docker-entrypoint-initdb.d/

ENV POSTGRES_USER=postgres
ENV POSTGRES_PASSWORD=postgres
ENV POSTGRES_DB=lightrag
