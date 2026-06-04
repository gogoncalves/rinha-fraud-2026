# rinha-fraud-2026

Detecção de fraude KNN-5 em vetores 14-D, escrito em Zig. Submissão para [Rinha de Backend 2026](https://github.com/zanfranceschi/rinha-de-backend-2026).

## Stack

- **Zig 0.14.1**, cross-compile pra `x86_64-linux-musl` + `-mcpu=haswell`
- **IVF k=4096** + k-means++ (12 iters, sample 400k)
- **int16** quantizado (scale=4096)
- **SoA 8-lane pair-packed** + AVX2 `VPMADDWD`
- **Early-prune** em 6/14 dims via `cmpgt` (skipa block inteiro se todas as 8 lanes já passaram do 5º-melhor)
- **Repair** sobre top-32 clusters por distância de centroide com filtro bbox
- HTTP server hand-rolled em `epoll` ET single-thread, `EPIOCSPARAMS` busy-poll (Linux 6.0+)
- LB próprio em Zig: `accept()` TCP + `SCM_RIGHTS` sendmsg pra entregar fd aos workers via UDS SEQPACKET
- Index mmap + `MADV_HUGEPAGE/WILLNEED/RANDOM` + prefault + `mlockall`
- HTTP responses pre-renderizados (6 estados: 0.0/0.2/0.4/0.6/0.8/1.0)

## Arquitetura

```
client → :9999 lb → SCM_RIGHTS fd → api1 / api2 (round-robin)
```

| serviço | cpuset | CPU  | RAM    |
|---------|--------|------|--------|
| lb      | 2,3    | 0.30 | 20 MB  |
| api1    | 0      | 0.35 | 165 MB |
| api2    | 1      | 0.35 | 165 MB |

Total: 1.0 CPU / 350 MB.

## Como roda

```sh
docker compose up -d
curl http://localhost:9999/ready
curl -X POST http://localhost:9999/fraud-score \
  -H 'content-type: application/json' \
  --data @rinha-test/sample.json
```

## Build

A imagem é construída em multi-stage:

1. baixa `references.json.gz` (3M vetores)
2. roda `build-index`: k-means++ K=4096 (sample 400k), atribui os 3M vetores, calcula bbox por cluster, escreve `index.bin` (~96 MB) no layout SoA pair-packed
3. compila `api` e `lb` (~90 KB cada, stripped)

```sh
docker compose build
```

Local sem Docker: `make` (requer `zig` 0.14).

## Layout

```
src/
  main.zig       epoll loop, UDS SEQPACKET ou TCP, fd-passing, busy-poll
  lb.zig         TCP accept + SCM_RIGHTS round-robin pros backends
  http.zig       parser POST /fraud-score + GET /ready
  json.zig       descida recursiva, schema fixo, zero-alloc
  normalize.zig  14 dims, clamp, MCC risk packed-u32 switch
  index.zig     mmap, IVF query, SoA SIMD scan com early-prune, repair
  time.zig       ISO-8601 → epoch (Hinnant), dia da semana (Sakamoto)
tools/
  build_index.zig  parse JSON → k-means++ → bucketize → bin
```

## Licença

MIT.
