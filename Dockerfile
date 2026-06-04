FROM --platform=linux/amd64 alpine:3.20 AS builder

ARG ZIG_VERSION=0.14.1
RUN apk add --no-cache curl tar xz ca-certificates gzip gcc musl-dev \
 && curl -fsSL "https://ziglang.org/download/${ZIG_VERSION}/zig-x86_64-linux-${ZIG_VERSION}.tar.xz" \
    | tar -xJ -C /opt \
 && ln -s /opt/zig-x86_64-linux-${ZIG_VERSION}/zig /usr/local/bin/zig

WORKDIR /src

COPY tools ./tools
RUN mkdir -p zig-out \
 && zig build-exe -O ReleaseFast -lc -fstrip \
      -target x86_64-linux-musl -mcpu haswell \
      -femit-bin=zig-out/build-index tools/build_index.zig

ARG REFERENCES_URL=https://github.com/zanfranceschi/rinha-de-backend-2026/raw/main/resources/references.json.gz
RUN curl -fsSL -o /tmp/refs.json.gz "${REFERENCES_URL}" \
 && gunzip /tmp/refs.json.gz \
 && BUILD_KD=1 INPUT=/tmp/refs.json OUTPUT=/index.bin /src/zig-out/build-index \
 && rm /tmp/refs.json

COPY src ./src
# Note: no -fsingle-threaded — main.zig spawns std.Thread workers when
# API_WORKERS>1. With API_WORKERS=1 the threading scaffolding is unused.
RUN zig build-exe -O ReleaseFast -lc -fstrip \
      -target x86_64-linux-musl -mcpu haswell \
      -femit-bin=zig-out/api src/main.zig \
 && gcc -O3 -static -s -march=haswell -o zig-out/lb src/lb.c

FROM --platform=linux/amd64 scratch
COPY --from=builder /src/zig-out/api /api
COPY --from=builder /src/zig-out/lb /lb
COPY --from=builder /index.bin /data/index.bin
ENV INDEX_PATH=/data/index.bin
EXPOSE 9999
ENTRYPOINT ["/api"]
