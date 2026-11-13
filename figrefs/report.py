"""Self-contained HTML review page per paper: figure image, caption, explicit paragraphs, implicit candidates."""
from __future__ import annotations

import base64
import html
import re
from pathlib import Path


def _hl(text: str, needles: list[str]) -> str:
    out = html.escape(text)
    for n in sorted(set(needles), key=len, reverse=True):
        out = out.replace(html.escape(n), f"<mark>{html.escape(n)}</mark>")
    return out


def write_report(result: dict, doc, pdir: Path) -> None:
    parts = [f"<h1>{html.escape(result['paper'].get('title') or result['paper']['file'])}</h1>",
             f"<p class=meta>{html.escape(result['paper']['file'])} · {len(result['figures'])} figures · "
             f"{result['n_paragraphs']} paragraphs</p>"]
    for f in result["figures"]:
        img = ""
        if f["image"]:
            b64 = base64.b64encode((pdir / f["image"]).read_bytes()).decode()
            img = f"<img src='data:image/png;base64,{b64}'>"
        exp_html = []
        for e in f["explicit"]:
            sent_texts = [m["text"] for m in e["mention_sentences"]]
            exp_html.append(f"<div class=para><span class=tag>¶{e['paragraph_id']} · p.{','.join(map(str, e['pages']))} · "
                            f"{html.escape(e['section'][:60])}</span><p>{_hl(e['paragraph'], sent_texts)}</p></div>")
        cap_html = "".join(f"<div class=para><span class=tag>caption of Fig. {c['from_figure']}</span>"
                           f"<p>{html.escape(c['caption'])}</p></div>" for c in f["caption_mentions"])
        imp_html = []
        for i in f["implicit"][:12]:
            sig = ", ".join(f"{k}={v}" for k, v in i["signals"].items())
            cls = "sel" if i["selected"] else "cand"
            imp_html.append(f"<div class='imp {cls}'><span class=score>{i['score']:.2f}</span>"
                            f"<span class=tag>¶{i['paragraph_id']}.{i['sentence_index']} · {html.escape(sig)}</span>"
                            f"<p>{html.escape(i['text'])}</p></div>")
        parts.append(f"""
<section>
  <h2>Figure {f['figure_id']} <small>page {f['page']}</small></h2>
  <div class=grid>
    <div class=fig>{img}<p class=cap>{html.escape(f['caption'])}</p></div>
    <div>
      <h3>Explicit paragraphs ({len(f['explicit'])})</h3>{''.join(exp_html) or '<p class=none>none</p>'}
      {('<h3>Mentioned in other captions</h3>' + cap_html) if cap_html else ''}
      <h3>Implicit candidates ({sum(i['selected'] for i in f['implicit'])} selected / {len(f['implicit'])})</h3>
      {''.join(imp_html) or '<p class=none>none</p>'}
    </div>
  </div>
</section>""")
    page = f"""<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>figrefs · {html.escape(result['paper']['file'])}</title>
<style>
:root{{--bg:#fbfaf7;--fg:#1d1d1b;--muted:#6b6a65;--card:#fff;--line:#e4e2dc;--mark:#ffe58a;--sel:#d9f2e3;--cand:#f3f2ee;--acc:#2f6f4f}}
@media (prefers-color-scheme:dark){{:root{{--bg:#171716;--fg:#ecebe6;--muted:#a3a29c;--card:#211f1e;--line:#353330;--mark:#6b5a12;--sel:#1f3b2c;--cand:#262523;--acc:#7fc8a0}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,sans-serif;margin:0;padding:24px 16px;max-width:1300px;margin:auto}}
h1{{font-size:22px;margin:0 0 4px}} .meta{{color:var(--muted);margin:0 0 24px}}
section{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:18px;margin-bottom:22px}}
h2{{margin:0 0 12px;font-size:18px}} h2 small{{color:var(--muted);font-weight:400;font-size:13px}}
h3{{font-size:13px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);margin:16px 0 8px}}
.grid{{display:grid;grid-template-columns:minmax(0,5fr) minmax(0,7fr);gap:22px}}
@media (max-width:800px){{.grid{{grid-template-columns:1fr}}}}
.fig img{{max-width:100%;border:1px solid var(--line);border-radius:6px;background:#fff}}
.cap{{font-size:13px;color:var(--muted)}}
.para p,.imp p{{margin:4px 0 0}} .para,.imp{{border-left:3px solid var(--line);padding:6px 10px;margin-bottom:10px}}
.imp.sel{{background:var(--sel);border-color:var(--acc)}} .imp.cand{{background:var(--cand)}}
.tag{{font-size:12px;color:var(--muted)}} .score{{font-weight:600;margin-right:8px;font-variant-numeric:tabular-nums}}
mark{{background:var(--mark);color:inherit;padding:0 1px}} .none{{color:var(--muted)}}
</style></head><body>{''.join(parts)}</body></html>"""
    (pdir / "report.html").write_text(page)
