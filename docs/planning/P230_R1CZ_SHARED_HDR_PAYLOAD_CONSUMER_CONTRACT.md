# P230 — R1CZ persisted shared-HDR payload consumer audit

## Question

Can this repository independently hash, canonically deserialize, validate and
apply the exact 4,376-byte R1CZ artifact using only the persisted payload and
schema/envelope facts, without rebuilding the paired reference bundle?

## Frozen artifacts and fixture

- Producer evidence commit: `289e32b5a972489533dba253ede1f91213bcafb3`.
- Formal producer execution commit: `70d235b0f4e4a276c4ef7dace306e8b42678f524`.
- Payload: exact tracked blob, 4,376 bytes, SHA-256
  `aa7fe0400e7bef106a9ff75305babab317a4148bdb05c573c50395d6f4228cb9`.
- The consumer fixture is the float32 Cartesian product of
  `[-1, 0, .001, 1, 18, 203, 1000, 10000, 12000]` in RGB, reshaped to
  `729x1x3`. Its byte SHA-256 is
  `6476554cf17e8c47bd961d61362a17a3443d1eab15826aa1dc1326fb6a0532ec`.
- The producer oracle is the exact committed portable float32 interpreter;
  it runs in a separate process and receives only the persisted payload plus
  this consumer fixture. It does not fit or rebuild a bundle.

## Gates

The consumer must verify evidence, blob, byte length, canonical UTF-8 JSON,
schema-valid envelope, payload hash, bundle ID and strict format. Consumer and
producer-oracle float32 outputs must be byte exact; inputs remain unchanged,
outputs are owned/contiguous/finite and clipped only by the frozen payload
contract to `[0,10000]`. Corrupted bytes, wrong envelope/hash and unsupported
format fail before output. Forward/reverse reports must be byte exact.

## Stop rule and claim ceiling

Any failed gate closes the consumer without rebuilding the paired bundle,
changing arithmetic, tolerances, fixture, payload or envelope. A pass is only
private compatibility for this exact self-authored R1CY/R1CZ payload and the
frozen consumer fixture. It does not admit arbitrary payloads, after-only
inference, natural/captured HDR quality, public package/schema/capability,
display/media quality or product behavior.
