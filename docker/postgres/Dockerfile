# PostgreSQL 18 with pgvector (vector search, included in the base image)
# and pg_textsearch (BM25 ranking, https://github.com/timescale/pg_textsearch).
FROM pgvector/pgvector:0.8.7-pg18-trixie

# Official pg_textsearch release for PostgreSQL 18. The build fails if the checksum differs.
ADD --checksum=sha256:465fe95726b4f26d354cadeea1ba983441f239ad75013bd54038dabdf2677101 \
    https://github.com/timescale/pg_textsearch/releases/download/v1.5.1/pg_textsearch-pg18-v1.5.1.tar.gz \
    /tmp/pg_textsearch.tar.gz

# Copy the library and the extension files to where PostgreSQL looks for them.
RUN tar -xzf /tmp/pg_textsearch.tar.gz -C /tmp \
    && cp /tmp/pg18/pg_textsearch.so /usr/lib/postgresql/18/lib/ \
    && cp /tmp/pg18/pg_textsearch.control /tmp/pg18/pg_textsearch--*.sql /usr/share/postgresql/18/extension/ \
    && rm -rf /tmp/pg18 /tmp/pg_textsearch.tar.gz

# pg_textsearch must be loaded at server start, otherwise CREATE EXTENSION fails.
CMD ["postgres", "-c", "shared_preload_libraries=pg_textsearch"]
