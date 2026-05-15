# ---- Builder ----
FROM rust:1.85-bookworm AS builder

WORKDIR /app
COPY Cargo.toml Cargo.lock* ./
COPY src/ src/

RUN cargo build --release --bin dhara-pizza

# ---- Runtime ----
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libssl3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/dhara-pizza /usr/local/bin/dhara-pizza

ENV PORT=10000
EXPOSE 10000

CMD ["dhara-pizza"]
