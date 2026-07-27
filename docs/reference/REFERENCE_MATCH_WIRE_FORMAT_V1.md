# Reference Match Wire Identity v1

## Scope

This document defines the language-neutral byte stream used to verify
`recipe_id` and `plan_id`. JSON files remain the transport format and are
validated by the corresponding Draft 2020-12 schemas. Identity hashes are not
computed from JSON text because whitespace, key order and float formatting vary
across platforms.

## Canonical value subset

Only these parsed JSON-compatible values are supported:

- null;
- boolean;
- arbitrary signed integer;
- finite IEEE-754 binary64 number;
- Unicode string;
- ordered array;
- object whose keys are Unicode strings.

NaN, infinities, byte strings and non-string object keys are invalid.

## Encoding

All tags and lengths are ASCII. String payloads are UTF-8. `LEN:` is the
unsigned decimal byte or item count followed by `:`.

| Value | Bytes |
|---|---|
| null | `n;` |
| false / true | `b0;` / `b1;` |
| integer `N` | `i` + minimal signed decimal `N` + `;` |
| binary64 float `F` | `f` + 16 lowercase hexadecimal digits of the big-endian IEEE-754 bit pattern + `;` |
| string `S` | `s` + UTF-8 byte `LEN:` + UTF-8 bytes |
| array | `l` + item `LEN:` + canonical bytes of every item in order |
| object | `d` + member `LEN:` + each key as a canonical string followed by its value |

Object keys are sorted by Unicode scalar value before encoding. For valid
Unicode strings this order is also preserved by UTF-8 byte ordering.

Integer and float are different types. Positive and negative zero have
different binary64 bit patterns and therefore different identities.

## Hash construction

1. Parse and validate the JSON payload.
2. For a reference recipe, remove the top-level `recipe_id`.
3. For a composition plan, remove the top-level `plan_id`.
4. Encode the remaining object using the canonical rules above.
5. Compute SHA-256 over the exact canonical bytes.

Promotion-report identity follows the same byte grammar. Local absolute path
maps are recorded for provenance but excluded from `report_id`; the report
instead binds reference, source and target file SHA-256 maps. Moving identical
evidence between Windows, macOS, Android or iOS storage therefore does not
change the content-bound report identity.

The product run report binds `reference-render-guard.v2`. Every safety row
records `research_baseline_override`. When it is false, the row must be
`identity-fallback` and include `algorithm-not-promoted`; when true, that
reason is removed but gamut and new-boundary vetoes remain authoritative.
6. Encode the digest as 64 lowercase hexadecimal characters.

The run report carries recipe identity but does not have a recursive internal
report ID; the report file itself is identified by its normal file SHA-256.

## Frozen vector

Parsed value:

```json
{"a":[null,true,-2,1.5,"色"],"z":-0.0}
```

Canonical bytes, with the UTF-8 string shown as escaped bytes:

```text
d2:s1:al5:n;b1;i-2;f3ff8000000000000;s3:\xe8\x89\xb2s1:zf8000000000000000;
```

SHA-256:

```text
283b5aeb8ea476e896c26c821b83812976c7afa8ac30f1d87a70fcb6687188a5
```

Platform implementations must pass this vector before they may claim recipe or
composition identity parity.
