import re

with open(r'c:\Users\Ravi\Downloads\Project\index.html', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Add CSS overrides
css_overrides = """
/* Light theme overrides for Detection Results */
body.light-theme .result-hero { background: linear-gradient(135deg, rgba(99,102,241,0.04), rgba(56,189,248,0.02)); }
body.light-theme .export-bar { background: rgba(0,0,0,0.03); }
body.light-theme .emo-track { background: rgba(0,0,0,0.06); }
body.light-theme .dash-panel { background: rgba(255,255,255,0.7); box-shadow: 0 4px 32px rgba(0,0,0,0.06), inset 0 1px 0 rgba(255,255,255,0.5); }
body.light-theme .dash-title { color: rgba(99,102,241,0.8); }
body.light-theme .rl-item { color: var(--text2); }
body.light-theme .conf-item { background: rgba(0,0,0,0.02); }
body.light-theme .conf-item:hover::before { background: rgba(0,0,0,0.02); }
body.light-theme .conf-item.top { background: rgba(99,102,241,0.05); }
body.light-theme .conf-emoji-wrap { background: rgba(0,0,0,0.04); }
body.light-theme .conf-label { color: var(--text); }
body.light-theme .conf-item.top .conf-label { color: var(--p1); }
body.light-theme .conf-pct { color: var(--text3); }
body.light-theme .conf-item.top .conf-pct { color: var(--p1); }
body.light-theme .conf-bar-track { background: rgba(0,0,0,0.06); }
body.light-theme {
  --radar-bg: rgba(255,255,255,0.4);
  --radar-grid: rgba(0,0,0,0.1);
  --radar-grid-outer: rgba(0,0,0,0.15);
  --radar-tick: rgba(0,0,0,0.4);
  --radar-axis: rgba(0,0,0,0.05);
}
"""
if "/* Light theme overrides for Detection Results */" not in text:
    text = text.replace("</style>", css_overrides + "\n</style>")

# 2. Patch Radar Chart Javascript
text = text.replace("fill:'rgba(6,8,24,0.6)'", "fill:'var(--radar-bg, rgba(6,8,24,0.6))'")
text = text.replace("stroke:'rgba(99,130,255,0.08)'", "stroke:'var(--radar-grid, rgba(99,130,255,0.08))'")

text = text.replace("t===1 ? 'rgba(99,130,255,0.25)' : 'rgba(99,130,255,0.1)'", "t===1 ? 'var(--radar-grid-outer, rgba(99,130,255,0.25))' : 'var(--radar-grid, rgba(99,130,255,0.1))'")

text = text.replace("fill:'rgba(160,185,255,0.28)'", "fill:'var(--radar-tick, rgba(160,185,255,0.28))'")

text = text.replace("stroke:'rgba(99,130,255,0.12)'", "stroke:'var(--radar-axis, rgba(99,130,255,0.12))'")

with open(r'c:\Users\Ravi\Downloads\Project\index.html', 'w', encoding='utf-8') as f:
    f.write(text)

print("Patch applied.")
