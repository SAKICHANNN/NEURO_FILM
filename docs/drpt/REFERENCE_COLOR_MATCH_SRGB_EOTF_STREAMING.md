# Reference sRGB EOTF Streaming Evidence

Date: 2026-07-28

Status: **24 MP chunked Windows host execution is exact and bounded at the
declared array-payload level; other target runtimes remain open**.

## Scope

P86 audits the consumer-owned P82/P84 decoded-sample EOTF ABI under a
large-image calling pattern. It does not add a media decoder, allocate an
image-sized native scratch buffer, implement a D-PCT transform or admit an
algorithm.

The frozen workload is:

- 24,000,000 RGB pixels / 72,000,000 scalar samples;
- both uint8 and native-endian uint16 input;
- 1,048,579 scalar samples per chunk, yielding 69 chunks;
- deterministic full-range modular sample patterns;
- two executions per bit depth and compiler;
- independent NumPy/P72 lookup comparison for every output scalar;
- MSVC and LLVM-MinGW Windows x86_64 DLLs built from the same P84 source.

## Exact results

Two complete top-level audits independently produced the same stable evidence
ID:

`sha256:5f7a5150d65e121f0f68f1d55a6edb4051296d65694cb0af01d24f56fcfa579a`

The timing-bearing report files differ as intended:

| Run | Report SHA-256 |
|---|---|
| A | `df3f6f5fa7ae5593b7385637a951d7f6f357132936b8bfb3631c9d7fdaed9c18` |
| B | `b2e897db5ad5d4799dafda5187805832c3423dbaa618023a0228759bdf9bd027` |

Their complete non-timing `evidence` objects are equal. Within each audit,
both replays and both compiler implementations are equal.

| Depth | Input SHA-256 | Output float32-byte SHA-256 | Chunks |
|---:|---|---|---:|
| 8 | `465bef3000b6c54ef1cc0eef2933cc3afed8baa8c64ce8f3453789c2dc35a47c` | `994c30d3049808893510e23bfa50500dbf43909683435bbbdaf1c5ec184aa12c` | 69 |
| 16 | `2ebfe96dc0c2e905a4e44d1c0be122ccdf7da450f70f58da55e78790675baba0` | `ad7007c3db8e35c97b6ce9fde452ebc232a2a05124e719f7f0173fbe81a3f148` | 69 |

## Resource boundary

The native ABI has no internal allocator and consumes caller-owned buffers.
The auditor tracks the exact NumPy payload bytes simultaneously live for the
source, expected oracle and native output arrays:

| Depth | Maximum live tracked array payload |
|---:|---:|
| 8 | 9,437,211 bytes |
| 16 | 10,485,790 bytes |

This is not process RSS, allocator-overhead or mobile memory-pressure
evidence. It proves that the tested caller can process the 24 MP workload
without a full-image input, oracle or output allocation.

Observed native-call time per complete 72-million-sample pass across two
audits was approximately 0.043--0.070 seconds for uint8 and 0.062--0.094
seconds for uint16. End-to-end Python auditor time was approximately
0.66--0.86 seconds per pass. These are local observations, not an SLA.

## Failure and lifetime boundary

The auditor:

- rejects non-positive pixels/chunks and multiplication overflow;
- compares every chunk before incorporating it into the evidence hash;
- requires same-compiler replay equality and cross-compiler equality;
- separates timings from the stable evidence identity;
- releases every loaded Windows DLL in `finally`, including injected runtime
  failure, before temporary artifacts are removed;
- writes its final report create-only after the complete audit succeeds.

## Explicit non-claims

- No Android, iOS or macOS runtime was executed.
- No PNG/JPEG/TIFF/RAW/HDR/video byte was decoded.
- No ICC application, colour-match fit/apply, FilmFX, authorization,
  persistence or delivery occurred.
- The result does not change identity fallback, A1/A4/A5 or the producer
  compatibility boundary.
