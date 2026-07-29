# U6.P5F cross-layer interimage adjacency

Decision: **synthetic feasibility pass only**.

Two runs from software commit `666d854c` are byte-identical at report SHA-256
`317eed87...1251d`. The fixed zero-row-sum coupling:

- preserves all frozen constant fields and the complete neutral-axis edge
  exactly;
- increases peak opponent gradient by `6.8024%` on each single-channel edge;
- limits density and transmittance changes to `0.04999984` and
  `0.0060000001`;
- is exact at 31/47-row partitions, with zero response beyond its five-pixel
  finite halo and zero hard clipping.

This establishes only that a bounded cross-layer developed-density mechanism
can be represented safely on the frozen synthetic witnesses. It does not
identify real DIR chemistry, a stock/process response, photographic safety or
product value. P5G must test the unchanged operator on rights-cleared
photographic inputs with severe-artifact-first gates.
