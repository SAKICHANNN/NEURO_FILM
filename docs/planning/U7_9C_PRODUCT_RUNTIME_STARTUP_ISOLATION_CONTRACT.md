# U7.9C — Product Runtime Startup Isolation Contract

## Parent and observed defect

Parent: `ULT > U7 productization > U7.9C`.

U7.9A/U7.9B prove the exact private Windows CPython 3.12 runtime, but the
published command launcher starts Python without isolated mode.  A caller can
therefore supply `PYTHONPATH` or a user-site `sitecustomize` module that runs
before `product-launch.py` performs its repository and requirements checks.
The launcher also forwards that environment when it starts the product
renderer.  This is a source-lock and auditability defect in the existing
runtime, not a request for another installer or wrapper.

## Frozen repair

1. Keep the existing runtime layout, receipt schema, requirements, repository
   binding, product entry point and three Look Approximation outputs.
2. The command launcher must start `product-launch.py` with CPython isolated
   mode (`-I`).
3. `product-launch.py` must remove every inherited environment key whose name
   begins with `PYTHON` (case-insensitive), then start `render_film.py` with
   `-I` and the sanitized environment.
4. Repository HEAD, tracked-clean and requirements-SHA checks remain before
   the renderer subprocess.  This leaf does not broaden those checks into a
   general hostile-host or executable-provenance claim.
5. Existing destinations, installation failure cleanup and receipt semantics
   remain unchanged.

## Frozen controls and gates

- a hostile `PYTHONPATH` containing `sitecustomize.py` must create no marker
  before or during catalog execution;
- mixed-case `PYTHON*` variables must not reach the renderer subprocess;
- the launcher source and command bytes must bind isolated mode explicitly;
- catalog JSON and all three tiny product render bytes must remain exact to an
  uncontaminated direct installed-interpreter oracle;
- requirements drift, repository drift, invalid look, source immutability and
  installation ownership controls remain fail-closed;
- forward/reverse committed-head reports must be byte-identical after only
  frozen timing normalization.

## Claim ceiling and stop rule

A pass establishes only Python startup-environment isolation for the existing
private Windows repository-bound `film-inspired / Look Approximation` runtime.
It is not a sandbox, hostile-host guarantee, signed executable, standalone or
public installer, release approval, calibrated stock response or physical-film
reproduction.  Do not extend this leaf into PATH/DLL hardening, packaging,
installer variants or public distribution.
