# P231 — exact R1CZ payload on consumed real HDR video

## Question

Does the exact persisted R1CZ paired/capture-time shared HDR payload remain
finite, boundary-safe, material and temporally non-amplifying when it is
applied unchanged to the already-consumed R0VC native PQ/BT.2020 video?

## Frozen information flow

- The candidate is exactly the P230-verified R1CZ payload and envelope. Build
  reads no application frame and performs no fitting, routing or adaptation.
- The application is exactly the producer R0VC 300-frame file already consumed
  by R1CT. This leaf opens no fresh or sealed video role.
- The installed FFmpeg 8.1 runtime decodes normalized `gbrpf32le` PQ codes.
  Rows and columns are sampled at stride 8 and converted analytically to
  absolute display-linear nits. The sampled source float32 SHA must equal the
  exact producer R1CT/PyAV source SHA before candidate safety is interpreted.
- The same frozen payload is applied independently to every sampled frame.
  There is no target/reference grade, output media or network access.

## Frozen metrics and gates

Use the exact R1CT coordinates and thresholds: BT.2020 luminance in
`log2(1+nits)`, two log-ratio chroma coordinates, adjacent-frame absolute
steps, median log-RGB material effect, strict-interior new boundary and local
luminance spikes. Require finite output in `[0,10000]`, zero new boundaries,
material effect at least `.01`, luminance/chroma temporal p95 ratios at most
`1.25`, and new luminance-spike fraction at most `.005`. Forward-frame and
reverse-frame reductions must produce identical scientific payloads.

## Stop rule and claim ceiling

Any identity, decode-domain, range, boundary, materiality, temporal, spike or
replay failure closes this exact payload/video application. Do not change
payload strength/knots, frame set, stride, decoder, colour conversion,
threshold or gate after execution. A pass is one private non-commercial
target-blind temporal-safety D0 for this exact paired payload and consumed
video only. It does not establish grade correctness, aesthetic quality,
arbitrary video, natural paired HDR quality, a public package/schema/capability
or product admission.
