# U7.10B Installed-Runtime Desktop Launch Contract

## Role

`U7.10B` is the single installed-runtime composition leaf opened by the passed
`U7.10A` new-input desktop workflow. It is not permission to resume adjacent
bootstrap, PATH, DLL, packaging, cache, browser, or generic launcher variants.

## Question

Can one fresh private `U7.9A`-style Windows CPython 3.12 runtime expose both
the unchanged command-line renderer and the unchanged `U7.10A` native desktop
workflow from a foreign current directory, while preserving source identity,
startup isolation, explicit selection, create-only installation, and the
existing Look Approximation claim ceiling?

## Frozen implementation boundary

- `scripts/install_product_runtime.py` remains the only installer. It may add
  one desktop launcher and version its new receipt, but it must not copy or
  reimplement the renderer, preview path, UI, product catalog, or colour
  algorithm.
- The installed distribution set remains exactly `requirements-product-v2.txt`.
  No GUI, packaging, automation, or testing dependency may be added.
- `kmcfm-look.cmd` continues to invoke the existing isolated CLI launcher.
  Its generated source semantics and all three tiny direct-versus-launcher
  product outputs must remain exact.
- `kmcfm-desktop.cmd` invokes a separately generated Python launcher under the
  installed interpreter with `-I`. The generated launcher performs the same
  repository directory, requirements SHA, bound Git commit, and tracked-clean
  checks before executing `scripts/open_product_desktop.py`.
- Both generated launchers remove every inherited case-insensitive `PYTHON*`
  variable from their child environment. Source/requirements/tracked drift
  must reject before importing `tkinter`, creating a window, or reading an
  input image.
- The new receipt schema binds the path and SHA-256 identity of all four
  generated executable-chain files: `kmcfm-look.cmd`, `product-launch.py`,
  `kmcfm-desktop.cmd`, and `product-desktop-launch.py`. It retains the existing
  legacy CLI `launcher` field for current private consumers. Historical v1
  receipt/evidence stays immutable and is verified from its recorded Git object
  rather than relabelled as current.
- Installation remains create-only and repository-bound. Existing or
  late-foreign destinations remain unchanged; cleanup removes only a still
  identity-owned failed installation.

## Success gates

1. A fresh exact runtime installs with the same pinned distributions and
   `pip check` result as U7.9A.
2. The versioned receipt binds all four exact command/Python launcher files,
   source commit, requirements identity, repository path, claim and private
   status.
3. From a foreign current directory, the CLI launcher reproduces the direct
   catalog and all three deterministic Look Approximation outputs exactly.
4. From a foreign current directory and hostile mixed-case `PYTHON*`
   environment, the desktop launcher starts the real Tk `mainloop` under the
   installed interpreter without executing injected startup code.
5. One real desktop run reaches preview-ready state, performs an actual radio
   `invoke`, records the selected look, and exits with zero owned scratch and no
   remaining owned process.
6. Source-commit, requirements, tracked-tree, invalid argument, existing
   destination, late-foreign destination and injected installer failure
   controls fail closed at their frozen boundaries.
7. Forward/reverse committed-head reports and the accepted visual state are
   exact and all U7.10A plus U7.9 behavioral parents remain green or are
   explicitly historical Git-object checks.

## Stop rule and claim ceiling

Any failure closes this exact integration without changing dependencies,
duplicating launch validation, weakening source checks, changing UI controls,
adding automation-only product flags, or opening another launcher variant.
Pass establishes only private repository-bound Windows installed-runtime
desktop launch mechanics for the existing deterministic `film-inspired / Look
Approximation` workflow. It does not establish a standalone/public installer,
signed executable, hostile-host sandbox, cross-platform GUI, calibrated stock
response, physical-film reproduction, stock distinguishability, population
preference, or release approval.

## Verification and rollback

- Commit this contract/config before implementation.
- Commit installer/tests separately, then run existing U7.9 and U7.10
  behavioral tests.
- Run formal forward/reverse only from a tracked-clean committed HEAD.
- Bind the accepted report, visual state and historical compatibility in a
  separate evidence commit before README/tracker/log propagation.
