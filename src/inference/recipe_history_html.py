"""Self-contained accessible HTML rendering for a recipe-history catalog."""

from __future__ import annotations

from html import escape
from typing import Any

from .recipe_history import RECIPE_HISTORY_SCHEMA_ID

RECIPE_HISTORY_HTML_SCHEMA_ID = "kmcfm.desktop-recipe-history-page.v1"


class RecipeHistoryHtmlError(ValueError):
    """Raised when a recipe-history catalog cannot be rendered safely."""


def _text(value: Any) -> str:
    if not isinstance(value, (str, int)) or isinstance(value, bool):
        raise RecipeHistoryHtmlError("catalog contains an unsupported text value")
    return escape(str(value), quote=True)


def _validate_catalog(catalog: dict[str, Any]) -> None:
    if set(catalog) != {"schema_id", "status", "counts", "entries"}:
        raise RecipeHistoryHtmlError("catalog keys differ")
    if catalog["schema_id"] != RECIPE_HISTORY_SCHEMA_ID:
        raise RecipeHistoryHtmlError("catalog schema differs")
    if catalog["status"] not in {"empty", "ready", "partial", "invalid"}:
        raise RecipeHistoryHtmlError("catalog status differs")
    counts = catalog["counts"]
    if not isinstance(counts, dict) or set(counts) != {
        "discovered",
        "valid",
        "invalid",
    }:
        raise RecipeHistoryHtmlError("catalog counts differ")
    if any(type(counts[key]) is not int or counts[key] < 0 for key in counts):
        raise RecipeHistoryHtmlError("catalog counts are invalid")
    entries = catalog["entries"]
    if not isinstance(entries, list) or counts["discovered"] != len(entries):
        raise RecipeHistoryHtmlError("catalog entry count differs")
    valid = 0
    invalid = 0
    for row in entries:
        if not isinstance(row, dict) or row.get("status") not in {"valid", "invalid"}:
            raise RecipeHistoryHtmlError("catalog row status differs")
        if row["status"] == "valid":
            valid += 1
            required = {
                "status",
                "recipe_path",
                "recipe_sha256",
                "profile_id",
                "profile_version",
                "input_path",
                "input_sha256",
                "input_color_state",
                "style",
                "seed",
                "enabled_effects",
                "output_path",
                "output_sha256",
                "output_format",
                "output_bit_depth",
                "output_label",
                "evidence_grade",
                "software_commit",
            }
            if set(row) != required:
                raise RecipeHistoryHtmlError("valid catalog row keys differ")
            if not isinstance(row["enabled_effects"], list) or any(
                not isinstance(value, str) for value in row["enabled_effects"]
            ):
                raise RecipeHistoryHtmlError("enabled effects differ")
            for key, value in row.items():
                if key not in {"enabled_effects", "status"}:
                    _text(value)
        else:
            invalid += 1
            if set(row) != {"status", "recipe_path", "error_code"}:
                raise RecipeHistoryHtmlError("invalid catalog row keys differ")
            _text(row["recipe_path"])
            _text(row["error_code"])
    if counts["valid"] != valid or counts["invalid"] != invalid:
        raise RecipeHistoryHtmlError("catalog valid/invalid counts differ")
    expected_status = (
        "empty"
        if not entries
        else "ready"
        if invalid == 0
        else "invalid"
        if valid == 0
        else "partial"
    )
    if catalog["status"] != expected_status:
        raise RecipeHistoryHtmlError("catalog aggregate status differs")


def _definition(label: str, value: Any, *, code: bool = False) -> str:
    rendered = _text(value)
    if code:
        rendered = f"<code>{rendered}</code>"
    return f"<div><dt>{escape(label)}</dt><dd>{rendered}</dd></div>"


def _valid_card(row: dict[str, Any]) -> str:
    enabled_effects = row["enabled_effects"]
    effects = ", ".join(enabled_effects) if enabled_effects else "None"
    search = " ".join(
        str(row[key])
        for key in (
            "recipe_path",
            "style",
            "input_path",
            "output_path",
            "output_format",
            "output_label",
            "evidence_grade",
        )
    ).lower()
    definitions = "".join(
        [
            _definition("Recipe", row["recipe_path"], code=True),
            _definition("Input", row["input_path"], code=True),
            _definition("Input SHA-256", row["input_sha256"], code=True),
            _definition("Output", row["output_path"], code=True),
            _definition("Output SHA-256", row["output_sha256"], code=True),
            _definition(
                "Encoding",
                f"{row['output_format']} · {row['output_bit_depth']}-bit",
            ),
            _definition("Effects", effects),
            _definition("Software commit", row["software_commit"], code=True),
        ]
    )
    return (
        f'<article class="recipe-card" data-recipe-card data-search="{escape(search, quote=True)}">'
        '<div class="card-heading">'
        f'<div><p class="eyebrow">{_text(row["profile_id"])} · v{_text(row["profile_version"])}</p>'
        f"<h2>{_text(row['style'])}</h2></div>"
        '<div class="badges" aria-label="Recipe claim">'
        f'<span class="badge">{_text(row["output_label"])}</span>'
        f'<span class="badge badge-muted">{_text(row["evidence_grade"])}</span>'
        "</div></div>"
        f'<dl class="recipe-details">{definitions}</dl>'
        "</article>"
    )


def _invalid_card(row: dict[str, Any]) -> str:
    search = f"{row['recipe_path']} {row['error_code']}".lower()
    return (
        f'<article class="recipe-card recipe-card-error" data-recipe-card data-search="{escape(search, quote=True)}">'
        '<div class="card-heading"><div><p class="eyebrow">Invalid recipe</p>'
        f"<h2>{_text(row['recipe_path'])}</h2></div>"
        '<span class="badge badge-error">fail-closed</span></div>'
        '<dl class="recipe-details">'
        f"{_definition('Reason', row['error_code'], code=True)}"
        "</dl></article>"
    )


def render_recipe_history_html(catalog: dict[str, Any]) -> bytes:
    """Render a validated catalog as deterministic self-contained UTF-8 HTML."""
    _validate_catalog(catalog)
    counts = catalog["counts"]
    cards = "".join(
        _valid_card(row) if row["status"] == "valid" else _invalid_card(row)
        for row in catalog["entries"]
    )
    if catalog["status"] == "empty":
        cards = (
            '<section class="empty-state" aria-labelledby="empty-title">'
            '<p class="eyebrow">History is ready</p><h2 id="empty-title">No recipes found</h2>'
            "<p>Render with recipe recording enabled, then refresh this local history document.</p>"
            "</section>"
        )
    warning = ""
    if counts["invalid"]:
        warning = (
            '<p class="notice" role="alert">'
            f"{counts['invalid']} recipe file"
            f"{'s' if counts['invalid'] != 1 else ''} failed validation and remain isolated."
            "</p>"
        )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; img-src 'none'; connect-src 'none'; font-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'">
  <title>Render history · K-MCFM</title>
  <style>
    :root {{ color-scheme: dark; --bg:#101210; --panel:#191c19; --panel-2:#202420; --line:#3a403a; --text:#f2f4ef; --muted:#aab2aa; --accent:#d7ff74; --danger:#ff9e91; --radius:18px; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; min-width:320px; background:radial-gradient(circle at top left,#20261d 0,#101210 38rem); color:var(--text); font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; line-height:1.5; }}
    .shell {{ width:min(1120px,calc(100% - 32px)); margin:0 auto; padding:48px 0 72px; }}
    .app-header {{ display:grid; grid-template-columns:minmax(0,1fr) auto; align-items:end; gap:24px; padding-bottom:28px; border-bottom:1px solid var(--line); }}
    h1,h2,p {{ margin-top:0; }} h1 {{ margin-bottom:8px; font-size:clamp(2rem,5vw,4.5rem); letter-spacing:-.055em; line-height:.98; }} h2 {{ margin-bottom:0; font-size:1.15rem; overflow-wrap:anywhere; }}
    .lede,.eyebrow {{ color:var(--muted); }} .lede {{ max-width:62ch; margin-bottom:0; }} .eyebrow {{ margin-bottom:6px; font-size:.72rem; font-weight:750; letter-spacing:.14em; text-transform:uppercase; }}
    .summary {{ display:flex; gap:10px; flex-wrap:wrap; justify-content:flex-end; }} .metric {{ min-width:92px; padding:12px 14px; border:1px solid var(--line); border-radius:14px; background:rgba(25,28,25,.8); }} .metric strong {{ display:block; font-size:1.4rem; }} .metric span {{ color:var(--muted); font-size:.78rem; }}
    .toolbar {{ display:grid; grid-template-columns:minmax(0,1fr) auto; gap:16px; align-items:end; padding:28px 0; }} label {{ display:block; margin-bottom:8px; font-weight:700; }} input[type=search] {{ width:100%; min-height:48px; border:1px solid var(--line); border-radius:14px; background:#0d0f0d; color:var(--text); padding:0 15px; font:inherit; }} input[type=search]::placeholder {{ color:#899188; }} input[type=search]:focus-visible {{ outline:3px solid var(--accent); outline-offset:3px; }}
    .result-status {{ margin:0 0 12px; color:var(--muted); }} .notice {{ padding:12px 14px; border-left:3px solid var(--danger); background:#2b1d1a; color:#ffd5cf; }}
    .recipe-grid {{ display:grid; gap:16px; }} .recipe-card {{ padding:22px; border:1px solid var(--line); border-radius:var(--radius); background:linear-gradient(145deg,var(--panel),var(--panel-2)); box-shadow:0 18px 40px rgba(0,0,0,.16); }} .recipe-card[hidden] {{ display:none; }} .recipe-card-error {{ border-color:#75483f; }}
    .card-heading {{ display:flex; justify-content:space-between; gap:20px; align-items:flex-start; }} .badges {{ display:flex; flex-wrap:wrap; justify-content:flex-end; gap:8px; }} .badge {{ display:inline-flex; align-items:center; min-height:28px; padding:4px 9px; border-radius:999px; background:var(--accent); color:#1c2410; font-size:.72rem; font-weight:800; }} .badge-muted {{ background:#343a33; color:var(--text); }} .badge-error {{ background:var(--danger); color:#32100b; }}
    .recipe-details {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px 22px; margin:22px 0 0; }} .recipe-details div {{ min-width:0; }} dt {{ margin-bottom:3px; color:var(--muted); font-size:.72rem; font-weight:750; letter-spacing:.06em; text-transform:uppercase; }} dd {{ margin:0; overflow-wrap:anywhere; }} code {{ color:#dfe8d8; font-family:"Cascadia Mono","SFMono-Regular",Consolas,monospace; font-size:.78rem; }}
    .empty-state {{ padding:56px 22px; border:1px dashed var(--line); border-radius:var(--radius); text-align:center; background:rgba(25,28,25,.65); }} #no-results {{ padding:24px; border:1px dashed var(--line); border-radius:var(--radius); color:var(--muted); text-align:center; }}
    .footer {{ margin-top:28px; padding-top:20px; border-top:1px solid var(--line); color:var(--muted); font-size:.82rem; }}
    @media (max-width:720px) {{ .shell {{ width:min(100% - 20px,1120px); padding-top:28px; }} .app-header,.toolbar {{ grid-template-columns:1fr; }} .summary {{ justify-content:flex-start; }} .recipe-details {{ grid-template-columns:1fr; }} .card-heading {{ display:grid; }} .badges {{ justify-content:flex-start; }} }}
    @media (prefers-reduced-motion:reduce) {{ *,*::before,*::after {{ scroll-behavior:auto!important; transition-duration:.001ms!important; animation-duration:.001ms!important; }} }}
  </style>
</head>
<body>
  <main class="shell">
    <header class="app-header">
      <div><p class="eyebrow">K-MCFM local workspace</p><h1>Render history</h1><p class="lede">Immutable recipe records for deterministic film-inspired look approximations. This page is read-only.</p></div>
      <div class="summary" aria-label="Catalog summary">
        <div class="metric"><strong>{counts["valid"]}</strong><span>Valid</span></div>
        <div class="metric"><strong>{counts["invalid"]}</strong><span>Invalid</span></div>
      </div>
    </header>
    <section aria-labelledby="history-tools-title">
      <div class="toolbar"><div><label id="history-tools-title" for="recipe-search">Search recipes</label><input id="recipe-search" type="search" autocomplete="off" placeholder="Style, path, format, or claim"></div><p class="eyebrow">Catalog: {_text(catalog["status"])}</p></div>
      {warning}
      <p id="result-status" class="result-status" role="status" aria-live="polite">{counts["discovered"]} recipes shown</p>
      <div class="recipe-grid" id="recipe-grid">{cards}</div>
      <p id="no-results" hidden>No matching recipes. Try a different search.</p>
    </section>
    <footer class="footer">Read-only recipe metadata · no pixels, network, telemetry, render, or export actions</footer>
  </main>
  <script>
    (() => {{
      const input = document.getElementById('recipe-search');
      const cards = Array.from(document.querySelectorAll('[data-recipe-card]'));
      const status = document.getElementById('result-status');
      const empty = document.getElementById('no-results');
      const update = () => {{
        const query = input.value.trim().toLocaleLowerCase();
        let visible = 0;
        for (const card of cards) {{
          const match = !query || card.dataset.search.includes(query);
          card.hidden = !match;
          if (match) visible += 1;
        }}
        status.textContent = `${{visible}} of ${{cards.length}} recipes shown`;
        empty.hidden = visible !== 0 || cards.length === 0;
      }};
      input.addEventListener('input', update);
      update();
    }})();
  </script>
</body>
</html>
"""
    return html.encode("utf-8")


__all__ = [
    "RECIPE_HISTORY_HTML_SCHEMA_ID",
    "RecipeHistoryHtmlError",
    "render_recipe_history_html",
]
