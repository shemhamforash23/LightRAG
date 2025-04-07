CREATE DATABASE lightrag;

CREATE EXTENSION IF NOT EXISTS vector;

SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';
