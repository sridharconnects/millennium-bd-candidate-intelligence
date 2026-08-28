"""Design system — premium SaaS shell (zinc light + indigo accent).

Not a dark theme swap: typography, spacing, surfaces, and chrome are product UI.
Interactive chrome uses indigo. Status colours stay semantic:
  verified / success — grounded
  abstained / warning — proposed then discarded
  missing / muted — never stated
  conflict / danger — disagreement
"""
from __future__ import annotations

import streamlit as st

# ---- tokens -----------------------------------------------------------------
INK = "#09090B"
MUTED = "#71717A"
FAINT = "#A1A1AA"
LINE = "#E4E4E7"
LINE_STRONG = "#D4D4D8"
SURFACE = "#FFFFFF"
ELEVATED = "#FAFAFA"
CANVAS = "#F4F4F5"
ACCENT = "#4F46E5"
ACCENT_SOFT = "#EEF2FF"
ACCENT_TEXT = "#3730A3"
SUCCESS = "#16A34A"
SUCCESS_SOFT = "#F0FDF4"
WARNING = "#D97706"
WARNING_SOFT = "#FFFBEB"
DANGER = "#DC2626"
DANGER_SOFT = "#FEF2F2"
SHADOW_SM = "0 1px 2px rgba(9,9,11,.04)"
SHADOW_MD = "0 4px 16px rgba(9,9,11,.06), 0 1px 2px rgba(9,9,11,.04)"
SHADOW_LG = "0 12px 40px rgba(9,9,11,.08), 0 2px 6px rgba(9,9,11,.04)"
RADIUS = "8px"
FONT = "'Plus Jakarta Sans',-apple-system,'Segoe UI',system-ui,sans-serif"
MONO = "'JetBrains Mono',ui-monospace,SFMono-Regular,Menlo,monospace"

STATUS = {
    "verified": ("#15803D", "#F0FDF4", "grounded in a verified source span"),
    "derived": ("#3730A3", "#EEF2FF", "computed in Python from verified fields"),
    "abstained": ("#B45309", "#FFFBEB", "a value was proposed but its quote could not be verified — discarded"),
    "conflicted": ("#B91C1C", "#FEF2F2", "rule and model disagree — routed to human review"),
    "human_corrected": ("#6D28D9", "#F5F3FF", "corrected by a reviewer"),
    "missing": ("#71717A", "#F4F4F5", "not present in the document"),
    "unverified": ("#71717A", "#F4F4F5", "present but unverified"),
}

SERIES = ["#4F46E5", "#16A34A", "#D97706", "#0EA5E9", "#DC2626", "#7C3AED",
          "#0891B2", "#CA8A04", "#71717A", "#DB2777"]

CSS = f"""
<style>
  @import url("https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,400,0,0&display=swap");

  :root {{
    --ink:{INK}; --muted:{MUTED}; --faint:{FAINT}; --line:{LINE};
    --line-strong:{LINE_STRONG}; --surface:{SURFACE}; --elevated:{ELEVATED};
    --canvas:{CANVAS}; --accent:{ACCENT}; --accent-soft:{ACCENT_SOFT};
    --accent-text:{ACCENT_TEXT}; --success:{SUCCESS}; --warning:{WARNING};
    --danger:{DANGER}; --r:{RADIUS}; --shadow-sm:{SHADOW_SM};
    --shadow-md:{SHADOW_MD}; --shadow-lg:{SHADOW_LG};
    --mm-chat-w:clamp(360px, 28vw, 460px);
    --font:{FONT}; --mono:{MONO};
  }}

  html, body, [class*="css"], .stApp {{
      font-family:var(--font) !important;
      color:{INK}; font-size:15px; letter-spacing:0;
      -webkit-font-smoothing:antialiased;
  }}
  .stApp {{
      background:
        linear-gradient(135deg, rgba(79,70,229,.045) 0%, transparent 38%),
        linear-gradient(225deg, rgba(14,165,233,.04) 0%, transparent 34%),
        linear-gradient(180deg, #F8FAFC 0%, {CANVAS} 44%, #EEF2F7 100%);
      background-attachment:fixed;
  }}
  header[data-testid="stHeader"] {{ display:none; }}
  div[data-testid="stToolbar"] {{ display:none; }}
  .block-container {{
      width:100%;
      max-width:min(1880px, calc(100vw - 1.25rem));
      padding:0.75rem clamp(0.8rem, 1.4vw, 1.65rem) 3rem;
  }}
  section[data-testid="stMain"] {{ overflow:auto; }}
  h1,h2,h3,h4 {{ font-weight:650; letter-spacing:0; color:{INK}; }}
  h1 {{ font-size:1.45rem; }} h2 {{ font-size:1.15rem; }} h3 {{ font-size:1.0rem; }}
  code, pre, .mm-mono {{ font-family:var(--mono) !important; }}

  /* ========== RAIL ========== */
  section[data-testid="stSidebar"] {{
      background:
        linear-gradient(180deg, rgba(255,255,255,.035), transparent 34%),
        linear-gradient(180deg, #09090B 0%, #18181B 52%, #111113 100%) !important;
      backdrop-filter:blur(16px);
      border-right:1px solid #27272A !important;
      min-width:248px; max-width:300px;
  }}
  section[data-testid="stSidebar"] .block-container {{
      padding:1.1rem 1.05rem 1.5rem;
  }}
  [data-testid="stSidebarCollapseButton"] {{ display:none !important; }}
  section[data-testid="stSidebar"][aria-expanded="false"] {{
      transform:none !important; visibility:visible !important;
  }}
  section[data-testid="stSidebar"] * {{ color:#E4E4E7; }}
  section[data-testid="stSidebar"] .mm-brand-name {{ color:#FAFAFA !important; }}
  section[data-testid="stSidebar"] .mm-brand-sub {{ color:#A1A1AA !important; }}
  section[data-testid="stSidebar"] .mm-brand-lockup {{
      border-bottom-color:#27272A !important;
  }}
  section[data-testid="stSidebar"] .mm-mark {{
      background:linear-gradient(145deg, #6366F1 0%, #4338CA 100%);
      box-shadow:0 6px 18px rgba(79,70,229,.35);
  }}
  section[data-testid="stSidebar"] .mm-rail-snapshot b {{ color:#FAFAFA !important; }}
  section[data-testid="stSidebar"] .mm-rail-snapshot span {{ color:#A1A1AA !important; }}
  section[data-testid="stSidebar"] .mm-nav-purpose {{
      background:linear-gradient(180deg, rgba(255,255,255,.07), rgba(255,255,255,.035));
      border-color:#2F2F36; color:#C4C4CC;
  }}
  section[data-testid="stSidebar"] .mm-rail-foot {{
      border-top:1px solid #27272A; color:#71717A;
  }}
  section[data-testid="stSidebar"] .st-key-rail_nav {{
      margin-top:12px;
  }}
  section[data-testid="stSidebar"] div[role="radiogroup"] label p {{
      color:#A1A1AA !important;
  }}
  section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {{
      color:#FAFAFA !important;
  }}
  section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-child(1)::before,
  section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-child(5)::before,
  section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-child(7)::before {{
      color:#71717A;
  }}
  section[data-testid="stSidebar"] div[data-testid="stExpander"] details {{
      background:rgba(255,255,255,.045) !important;
      border:1px solid #2F2F36 !important;
      border-radius:8px !important;
      box-shadow:inset 0 1px 0 rgba(255,255,255,.04) !important;
  }}
  section[data-testid="stSidebar"] div[data-testid="stExpander"] summary {{
      min-height:38px;
  }}
  section[data-testid="stSidebar"] div[data-testid="stExpander"] summary p {{
      color:#D4D4D8 !important;
      font-weight:650;
  }}
  section[data-testid="stSidebar"] div[data-testid="stExpander"] svg {{
      color:#A1A1AA !important;
  }}
  section[data-testid="stSidebar"] .stButton>button {{
      background:#27272A; color:#FAFAFA; border-color:#3F3F46;
  }}
  section[data-testid="stSidebar"] .stButton>button:hover {{
      background:#3F3F46; border-color:#52525B;
  }}

  .mm-brand-lockup {{
      display:flex; align-items:center; gap:11px;
      padding:4px 6px 16px 6px; margin-bottom:0;
      border-bottom:1px solid {LINE};
  }}
  .mm-mark {{
      width:34px; height:34px; border-radius:8px;
      background:linear-gradient(145deg, {INK} 0%, #3F3F46 100%);
      color:#fff; font-size:0.82rem; font-weight:700;
      display:flex; align-items:center; justify-content:center;
      letter-spacing:0;
      box-shadow:0 4px 12px rgba(9,9,11,.18);
  }}
  .mm-brand-name {{ font-size:0.92rem; font-weight:700; letter-spacing:0; }}
  .mm-brand-sub {{ font-size:0.68rem; color:{MUTED}; margin-top:1px; font-weight:500; }}

  .mm-rail-snapshot {{
      display:grid; grid-template-columns:repeat(3, minmax(0, 1fr));
      gap:7px; padding:12px 0 4px;
  }}
  .mm-rail-snapshot div {{
      min-width:0; padding:8px 7px;
      background:rgba(255,255,255,.045);
      border:1px solid #27272A; border-radius:8px;
      box-shadow:inset 0 1px 0 rgba(255,255,255,.035);
  }}
  .mm-rail-snapshot b {{
      display:block; font-size:0.88rem; line-height:1; font-weight:750;
      font-variant-numeric:tabular-nums;
  }}
  .mm-rail-snapshot span {{
      display:block; margin-top:4px; font-size:0.61rem; font-weight:650;
      text-transform:uppercase; letter-spacing:.06em;
      white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
  }}

  section[data-testid="stSidebar"] div[role="radiogroup"] {{ gap:3px; }}
  section[data-testid="stSidebar"] div[role="radiogroup"] label {{
      min-height:38px;
      padding:9px 10px 9px 13px; border-radius:8px; border:1px solid transparent;
      transition:background .18s ease, border-color .18s ease, box-shadow .18s ease,
                 transform .15s ease;
      cursor:pointer; position:relative;
      overflow:visible;
  }}
  section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{
      background:rgba(255,255,255,.06);
      border-color:#2F2F36;
      transform:translateX(2px);
  }}
  section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked),
  section[data-testid="stSidebar"] div[role="radiogroup"] label[data-selected="true"] {{
      background:linear-gradient(90deg, rgba(79,70,229,.32), rgba(79,70,229,.11));
      border-color:rgba(129,140,248,.55);
      box-shadow:0 8px 24px rgba(0,0,0,.18), inset 0 1px 0 rgba(255,255,255,.08);
      transform:none;
  }}
  section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked)::after,
  section[data-testid="stSidebar"] div[role="radiogroup"] label[data-selected="true"]::after {{
      content:""; position:absolute; left:-1px; top:7px; bottom:7px; width:3px;
      border-radius:0 3px 3px 0; background:#818CF8;
      box-shadow:0 0 18px rgba(129,140,248,.7);
  }}
  section[data-testid="stSidebar"] div[role="radiogroup"] label p {{
      font-size:0.84rem !important; font-weight:600; color:{MUTED};
      line-height:1.2;
      position:relative;
      padding-left:26px;
  }}
  section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p,
  section[data-testid="stSidebar"] div[role="radiogroup"] label[data-selected="true"] p {{
      color:#FAFAFA !important; font-weight:700;
  }}
  section[data-testid="stSidebar"] div[role="radiogroup"] label input[type="radio"],
  section[data-testid="stSidebar"] div[role="radiogroup"] [data-baseweb="radio"] > div:first-child,
  section[data-testid="stSidebar"] div[role="radiogroup"] label > div:has(input[type="radio"]),
  section[data-testid="stSidebar"] label[data-testid="stRadioOption"] > div > div > div:first-child,
  section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {{
      display:none !important;
  }}
  section[data-testid="stSidebar"] label[data-testid="stRadioOption"] > div,
  section[data-testid="stSidebar"] label[data-testid="stRadioOption"] > div > div {{
      width:100%;
      min-width:0;
  }}
  section[data-testid="stSidebar"] div[role="radiogroup"] label [data-testid="stWidgetLabel"] {{
      margin-left:0 !important;
  }}
  section[data-testid="stSidebar"] div[role="radiogroup"] label span[data-testid="stMarkdownContainer"] {{
      margin-left:0 !important;
  }}
  section[data-testid="stSidebar"] div[role="radiogroup"] label p::before {{
      position:absolute; left:0; top:50%; transform:translateY(-50%);
      width:18px; height:18px; display:inline-flex; align-items:center; justify-content:center;
      font-family:"Material Symbols Rounded";
      font-size:18px; line-height:1; font-weight:400;
      color:#8B8B96; font-feature-settings:"liga";
  }}
  section[data-testid="stSidebar"] div[role="radiogroup"] label[data-selected="true"] p::before,
  section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p::before {{
      color:#E0E7FF;
  }}
  section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-child(1) p::before {{ content:"search"; }}
  section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-child(2) p::before {{ content:"person"; }}
  section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-child(3) p::before {{ content:"star"; }}
  section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-child(4) p::before {{ content:"assignment"; }}
  section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-child(5) p::before {{ content:"upload_file"; }}
  section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-child(6) p::before {{ content:"flag"; }}
  section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-child(7) p::before {{ content:"monitoring"; }}
  section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-child(8) p::before {{ content:"hub"; }}
  section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-child(9) p::before {{ content:"account_tree"; }}
  section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-child(10) p::before {{ content:"settings"; }}

  section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-child(1),
  section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-child(5),
  section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-child(7) {{
      margin-top:28px;
  }}
  section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-child(1) {{
      margin-top:24px;
  }}
  section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-child(1)::before,
  section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-child(5)::before,
  section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-child(7)::before {{
      position:absolute; left:2px; top:-19px;
      display:block; font-size:0.62rem; font-weight:750; letter-spacing:.12em;
      text-transform:uppercase; color:{FAINT}; margin:0;
  }}
  section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-child(1)::before {{ content:"Workspace"; }}
  section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-child(5)::before {{ content:"Pipeline"; }}
  section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-child(7)::before {{ content:"Intelligence"; }}

  .mm-nav-purpose {{
      font-size:0.74rem; line-height:1.5; color:{MUTED};
      padding:10px 12px; margin:8px 0 4px;
      background:{ELEVATED}; border:1px solid {LINE}; border-radius:8px;
  }}
  .mm-nav-badge {{
      display:inline-flex; align-items:center; justify-content:center;
      min-width:18px; height:18px; padding:0 5px; margin-left:6px;
      border-radius:999px; font-size:0.65rem; font-weight:700;
      background:{ACCENT_SOFT}; color:{ACCENT_TEXT};
  }}
  .mm-rail-foot {{
      padding:16px 8px 4px; margin-top:8px;
      border-top:1px solid {LINE};
      display:flex; flex-direction:column; gap:8px;
  }}
  .mm-status {{
      display:inline-flex; align-items:center; gap:6px; width:fit-content;
      font-size:0.68rem; font-weight:650; letter-spacing:.03em;
      border-radius:999px; padding:4px 10px; border:1px solid {LINE};
      background:{SURFACE};
  }}
  .mm-status::before {{
      content:""; width:6px; height:6px; border-radius:50%;
  }}
  .mm-status-live {{ color:#15803D; background:{SUCCESS_SOFT}; border-color:#BBF7D0; }}
  .mm-status-live::before {{ background:{SUCCESS}; box-shadow:0 0 0 3px rgba(22,163,74,.2); }}
  .mm-status-demo {{ color:#B45309; background:{WARNING_SOFT}; border-color:#FDE68A; }}
  .mm-status-demo::before {{ background:{WARNING}; }}
  .mm-rail-meta {{
      font-size:0.66rem; color:{FAINT}; font-family:var(--mono);
  }}

  /* ========== TOP BAR ========== */
  .st-key-app_chrome {{
      position:sticky; top:0; z-index:50;
      background:rgba(244,244,245,.82);
      backdrop-filter:blur(18px) saturate(1.2);
      padding:8px 0 10px; margin:0 0 16px;
      border-bottom:1px solid {LINE};
  }}
  .mm-page-head {{ padding:2px 0; }}
  .mm-page-kicker {{
      font-size:0.65rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase;
      color:{FAINT}; margin-bottom:2px;
  }}
  .mm-page-title {{
      font-size:1.15rem; font-weight:700; letter-spacing:0; line-height:1.2;
  }}
  .mm-page-sub {{ font-size:0.78rem; color:{MUTED}; margin-top:3px; font-weight:450; }}
  .mm-crumb {{ color:{FAINT}; font-weight:550; }}

  .st-key-back_btn button, .st-key-home_btn button {{
      min-width:36px; height:34px; border-radius:8px; padding:0 11px;
      background:{SURFACE}; color:{INK}; border:1px solid {LINE};
      font-weight:600; box-shadow:var(--shadow-sm);
      transition:border-color .15s ease, box-shadow .15s ease, transform .12s ease;
  }}
  .st-key-back_btn button:hover:enabled, .st-key-home_btn button:hover {{
      border-color:{LINE_STRONG}; box-shadow:var(--shadow-md); transform:translateY(-1px);
  }}
  .st-key-back_btn button:disabled {{
      background:{ELEVATED}; color:{FAINT}; border-color:{LINE}; box-shadow:none;
  }}

  .st-key-kpi_band [data-testid="stPopover"],
  .st-key-kpi_band [data-testid="stPopover"] > div {{ width:auto; }}
  .st-key-kpi_band [data-testid="stPopoverButton"] {{
      min-height:34px; width:auto; padding:5px 12px; border-radius:999px;
      background:{SURFACE}; border:1px solid {LINE}; justify-content:center;
      box-shadow:var(--shadow-sm);
      transition:border-color .15s ease, box-shadow .15s ease, transform .12s ease;
  }}
  .st-key-kpi_band [data-testid="stPopoverButton"]:hover {{
      border-color:{ACCENT}; box-shadow:0 0 0 3px {ACCENT_SOFT}; transform:translateY(-1px);
  }}
  .st-key-kpi_band [data-testid="stPopoverButton"] p {{
      font-size:0.74rem; color:{MUTED}; text-transform:none; letter-spacing:0;
      line-height:1.2; margin:0; font-weight:500;
  }}
  .st-key-kpi_band [data-testid="stPopoverButton"] p strong {{
      display:inline; font-size:0.78rem; font-weight:700; color:{INK};
      text-transform:none; letter-spacing:0;
  }}
  .st-key-kpi_band [data-testid="stPopoverButton"] p em {{ display:none; }}
  .st-key-kpi_band [data-testid="stPopoverButton"] p span strong {{ color:inherit; }}

  /* ========== SURFACES ========== */
  .mm-panel {{
      background:{SURFACE}; border:1px solid {LINE}; border-radius:8px;
      padding:16px 18px; box-shadow:var(--shadow-sm);
  }}
  .mm-card {{
      background:{SURFACE}; border:1px solid {LINE}; border-radius:8px;
      padding:14px 16px; margin-bottom:10px; box-shadow:var(--shadow-sm);
      transition:border-color .15s ease, box-shadow .15s ease, transform .15s ease;
  }}
  .mm-card:hover {{
      border-color:{LINE_STRONG}; box-shadow:var(--shadow-md); transform:translateY(-1px);
  }}
  .mm-cand {{ display:flex; gap:14px; align-items:flex-start; }}
  .mm-cand-avatar {{
      flex:0 0 40px; width:40px; height:40px; border-radius:11px;
      background:linear-gradient(145deg, {ACCENT_SOFT}, #E0E7FF);
      border:1px solid #C7D2FE; color:{ACCENT_TEXT};
      font-weight:700; font-size:0.72rem; letter-spacing:.04em;
      display:flex; align-items:center; justify-content:center;
  }}
  .mm-cand-body {{ flex:1 1 auto; min-width:0; }}
  .mm-cand-score {{
      flex:0 0 auto; font-variant-numeric:tabular-nums;
      background:{ACCENT_SOFT}; border:1px solid #C7D2FE;
      color:{ACCENT_TEXT}; border-radius:999px; padding:3px 10px;
      font-size:0.72rem; font-weight:700;
  }}
  .mm-row {{ display:flex; gap:8px; align-items:baseline; flex-wrap:wrap; }}
  .mm-name {{ font-weight:650; font-size:0.95rem; color:{INK}; letter-spacing:0; }}
  .mm-sub {{ color:{MUTED}; font-size:0.8rem; }}
  .mm-mono {{ font-family:var(--mono); font-size:0.75rem; }}

  .mm-chip {{
      display:inline-block; padding:2px 8px; border-radius:999px;
      font-size:0.68rem; font-weight:600; margin:2px 4px 2px 0;
      border:1px solid transparent; white-space:nowrap;
  }}
  .mm-chip-plain {{ background:{ELEVATED}; color:{MUTED}; border-color:{LINE}; }}

  .mm-ev {{
      background:{WARNING_SOFT}; border-left:3px solid {WARNING};
      padding:8px 12px; font-size:0.8rem; border-radius:0 8px 8px 0;
      margin:6px 0; line-height:1.5;
  }}
  .mm-ev mark {{
      background:#FDE68A; color:#78350F; padding:1px 3px; border-radius:3px; font-weight:650;
  }}

  .mm-bar-wrap {{
      background:{ELEVATED}; border-radius:4px; height:7px; width:100%; overflow:hidden;
      border:1px solid {LINE};
  }}
  .mm-bar {{ height:7px; border-radius:4px; background:{ACCENT}; }}

  .mm-kpi {{
      background:{SURFACE}; border:1px solid {LINE}; border-radius:8px;
      padding:14px 16px; min-width:100px; margin-bottom:12px;
      box-shadow:var(--shadow-sm);
      transition:border-color .15s ease, box-shadow .15s ease;
  }}
  .mm-kpi:hover {{ border-color:{LINE_STRONG}; box-shadow:var(--shadow-md); }}
  .mm-kpi .v {{
      font-size:1.45rem; font-weight:700; letter-spacing:0; line-height:1.1;
      font-variant-numeric:tabular-nums;
  }}
  .mm-kpi .l {{
      color:{MUTED}; font-size:0.68rem; text-transform:uppercase;
      letter-spacing:0.07em; margin-top:6px; font-weight:650;
  }}
  .mm-kpi .h {{ color:{FAINT}; font-size:0.72rem; margin-top:5px; }}
  .mm-kpi .delta {{
      display:inline-flex; align-items:center; gap:3px; margin-top:8px;
      font-size:0.72rem; font-weight:650; border-radius:999px; padding:2px 7px;
  }}
  .mm-kpi .delta-up {{ background:{SUCCESS_SOFT}; color:#15803D; }}
  .mm-kpi .delta-down {{ background:{DANGER_SOFT}; color:#B91C1C; }}
  .mm-kpi .delta-flat {{ background:{ELEVATED}; color:{MUTED}; }}

  .mm-foot {{
      position:sticky; bottom:0; background:rgba(244,244,245,.92);
      backdrop-filter:blur(12px); border-top:1px solid {LINE};
      padding:8px 2px; font-size:0.7rem; color:{FAINT};
      font-family:var(--mono);
  }}

  .mm-banner {{
      background:{SUCCESS_SOFT}; border:1px solid #BBF7D0; color:#166534;
      border-radius:8px; padding:10px 14px; font-size:0.82rem; margin-bottom:12px;
  }}
  .mm-warn {{
      background:{WARNING_SOFT}; border:1px solid #FDE68A; color:#92400E;
      border-radius:8px; padding:10px 14px; font-size:0.82rem; margin-bottom:10px;
  }}
  .mm-danger {{
      background:{DANGER_SOFT}; border:1px solid #FECACA; color:#991B1B;
      border-radius:8px; padding:10px 14px; font-size:0.82rem; margin-bottom:10px;
  }}
  .mm-synth {{
      background:#F5F3FF; border:1px solid #DDD6FE; color:#5B21B6;
      border-radius:8px; padding:10px 14px; font-size:0.82rem; margin-bottom:10px; font-weight:600;
  }}
  .mm-empty {{
      text-align:center; padding:56px 24px; border:1px dashed {LINE_STRONG};
      border-radius:8px; background:{SURFACE}; color:{MUTED};
      box-shadow:var(--shadow-sm);
  }}
  .mm-empty .mm-empty-icon {{
      width:48px; height:48px; margin:0 auto 14px; border-radius:8px;
      background:{ELEVATED}; border:1px solid {LINE};
      display:flex; align-items:center; justify-content:center;
      font-size:1.2rem; color:{ACCENT};
  }}
  .mm-empty h4 {{ color:{INK}; font-size:1rem; margin:0 0 6px; font-weight:700; letter-spacing:0; }}

  .mm-toolbar {{
      display:flex; align-items:center; justify-content:space-between; gap:12px;
      flex-wrap:wrap; margin:4px 0 12px;
  }}
  .mm-toolbar-title {{
      font-size:0.95rem; font-weight:700; letter-spacing:0; color:{INK};
  }}
  .mm-toolbar-meta {{ font-size:0.75rem; color:{MUTED}; margin-top:2px; }}

  .mm-review-hero {{
      display:flex;
      align-items:flex-start;
      justify-content:space-between;
      gap:18px;
      padding:18px 18px 14px;
      margin-bottom:10px;
      border:1px solid {LINE};
      border-radius:8px;
      background:
        linear-gradient(135deg, rgba(79,70,229,.08), transparent 48%),
        linear-gradient(225deg, rgba(22,163,74,.06), transparent 42%),
        rgba(255,255,255,.92);
      box-shadow:var(--shadow-md);
      animation:mm-rise .34s cubic-bezier(.22,1,.36,1) both;
  }}
  .mm-review-flow {{
      display:flex;
      align-items:center;
      gap:8px;
      flex:0 0 auto;
      padding-top:4px;
      color:{MUTED};
      font-size:0.68rem;
      font-weight:750;
      text-transform:uppercase;
      letter-spacing:.06em;
  }}
  .mm-review-flow b {{
      display:block;
      width:34px;
      height:1px;
      background:{LINE_STRONG};
  }}
  .mm-review-metrics {{
      display:grid;
      grid-template-columns:repeat(6, minmax(0, 1fr));
      gap:9px;
      margin-bottom:12px;
  }}
  .mm-review-metrics div {{
      min-width:0;
      padding:11px 12px;
      border-radius:8px;
      border:1px solid {LINE};
      background:{SURFACE};
      box-shadow:var(--shadow-sm);
  }}
  .mm-review-metrics b {{
      display:block;
      color:{INK};
      font-size:1.2rem;
      line-height:1;
      font-weight:760;
      font-variant-numeric:tabular-nums;
  }}
  .mm-review-metrics span {{
      display:block;
      margin-top:7px;
      color:{MUTED};
      font-size:0.62rem;
      font-weight:760;
      text-transform:uppercase;
      letter-spacing:.07em;
      white-space:nowrap;
      overflow:hidden;
      text-overflow:ellipsis;
  }}
  .mm-review-metrics small {{
      display:block;
      margin-top:4px;
      color:{FAINT};
      font-size:0.69rem;
      line-height:1.25;
  }}
  .st-key-review_queue_panel,
  .st-key-review_case_panel,
  .st-key-review_context_panel {{
      background:{SURFACE} !important;
      border-radius:8px !important;
      box-shadow:var(--shadow-md) !important;
      animation:mm-rise .4s cubic-bezier(.22,1,.36,1) both;
  }}
  .st-key-review_case_panel {{
      overflow:hidden;
      background:
        linear-gradient(180deg, rgba(255,255,255,.96), rgba(255,255,255,.9)),
        linear-gradient(135deg, rgba(79,70,229,.055), transparent 44%) !important;
      font-size:0.96rem;
  }}
  .st-key-review_queue_panel input,
  .st-key-review_case_panel input {{
      font-size:0.92rem;
  }}
  .mm-panel-heading {{
      display:flex;
      justify-content:space-between;
      align-items:center;
      gap:10px;
      padding-bottom:9px;
      margin-bottom:10px;
      border-bottom:1px solid {LINE};
  }}
  .mm-panel-heading.compact {{
      margin-top:15px;
  }}
  .mm-panel-heading span {{
      color:{INK};
      font-size:0.9rem;
      font-weight:760;
      letter-spacing:0;
  }}
  .mm-panel-heading b {{
      color:{FAINT};
      font-family:var(--mono);
      font-size:0.64rem;
      font-weight:550;
      text-transform:uppercase;
      letter-spacing:.06em;
      white-space:nowrap;
  }}
  .mm-review-pick {{
      display:flex;
      gap:10px;
      align-items:flex-start;
      background:{SURFACE};
      border:1px solid {LINE};
      border-left:3px solid {LINE_STRONG};
      border-radius:8px;
      padding:10px 11px;
      margin:8px 0 5px;
      min-height:74px;
      box-shadow:var(--shadow-sm);
      transition:border-color .15s ease, background .15s ease, transform .15s ease,
                 box-shadow .15s ease;
  }}
  .mm-review-pick.warn {{ border-left-color:{WARNING}; }}
  .mm-review-pick.danger {{ border-left-color:{DANGER}; }}
  .mm-review-pick.ok {{ border-left-color:{SUCCESS}; }}
  .mm-review-pick.is-selected {{
      border-color:#C7D2FE;
      border-left-color:{ACCENT};
      background:{ACCENT_SOFT};
      box-shadow:0 0 0 3px rgba(79,70,229,.08), var(--shadow-sm);
      transform:translateY(-1px);
  }}
  .mm-review-avatar {{
      width:34px;
      height:34px;
      flex:0 0 34px;
      border-radius:8px;
      background:linear-gradient(145deg, {ACCENT} 0%, #7C3AED 100%);
      color:#fff;
      display:flex;
      align-items:center;
      justify-content:center;
      font-weight:760;
      font-size:0.72rem;
      box-shadow:0 6px 14px rgba(79,70,229,.2);
  }}
  .mm-review-avatar.xl {{
      width:54px;
      height:54px;
      flex-basis:54px;
      font-size:0.95rem;
  }}
  .mm-review-pick-body {{
      min-width:0;
      flex:1;
  }}
  .mm-review-pick-top {{
      display:flex;
      align-items:center;
      justify-content:space-between;
      gap:8px;
      color:{INK};
      font-size:0.82rem;
      font-weight:760;
      line-height:1.25;
  }}
  .mm-review-pick-top span {{
      white-space:nowrap;
      overflow:hidden;
      text-overflow:ellipsis;
  }}
  .mm-review-pick-top b {{
      flex:0 0 auto;
      color:{MUTED};
      background:rgba(255,255,255,.72);
      border:1px solid {LINE};
      border-radius:999px;
      padding:2px 7px;
      font-size:0.58rem;
      text-transform:uppercase;
      letter-spacing:.06em;
  }}
  .mm-review-pick-reason {{
      margin-top:5px;
      color:{MUTED};
      font-size:0.8rem;
      line-height:1.32;
      display:-webkit-box;
      -webkit-line-clamp:2;
      -webkit-box-orient:vertical;
      overflow:hidden;
  }}
  .mm-review-pick-action {{
      display:flex;
      align-items:center;
      gap:7px;
      margin-top:7px;
      padding:6px 8px;
      border-radius:8px;
      border:1px solid rgba(79,70,229,.16);
      background:rgba(238,242,255,.68);
      color:{ACCENT_TEXT};
      font-size:0.76rem;
      font-weight:700;
      line-height:1.25;
  }}
  .mm-review-pick-action b {{
      color:{ACCENT_TEXT};
      font-family:var(--mono);
      font-size:0.58rem;
      text-transform:uppercase;
      letter-spacing:.07em;
  }}
  .mm-review-pick-meta {{
      margin-top:7px;
      color:{FAINT};
      font-family:var(--mono);
      font-size:0.66rem;
      white-space:nowrap;
      overflow:hidden;
      text-overflow:ellipsis;
  }}
  .mm-review-empty {{
      padding:18px 12px;
      text-align:center;
      border:1px dashed {LINE_STRONG};
      border-radius:8px;
      color:{MUTED};
      background:{ELEVATED};
      font-size:0.8rem;
  }}
  .mm-review-case-head {{
      display:flex;
      gap:14px;
      align-items:flex-start;
      padding:14px 15px;
      margin-bottom:10px;
      border:1px solid {LINE};
      border-left:4px solid {ACCENT};
      border-radius:8px;
      background:{SURFACE};
      box-shadow:var(--shadow-sm);
  }}
  .mm-review-case-head.warn {{ border-left-color:{WARNING}; }}
  .mm-review-case-head.danger {{ border-left-color:{DANGER}; }}
  .mm-review-case-head.ok {{ border-left-color:{SUCCESS}; }}
  .mm-review-case-title {{
      min-width:0;
      flex:1;
  }}
  .mm-review-case-kicker {{
      color:{ACCENT_TEXT};
      font-size:0.64rem;
      font-weight:780;
      text-transform:uppercase;
      letter-spacing:.08em;
      margin-bottom:4px;
  }}
  .mm-review-case-title h2 {{
      margin:0;
      font-size:1.45rem;
      line-height:1.16;
      letter-spacing:0;
  }}
  .mm-review-case-title p {{
      margin:5px 0 0;
      color:{MUTED};
      font-size:0.98rem;
      line-height:1.4;
  }}
  .mm-review-source {{
      margin-top:5px;
      color:{FAINT};
      font-family:var(--mono);
      font-size:0.66rem;
      white-space:nowrap;
      overflow:hidden;
      text-overflow:ellipsis;
  }}
  .mm-review-severity {{
      align-self:stretch;
      min-width:78px;
      border-left:1px solid {LINE};
      padding-left:14px;
      display:flex;
      flex-direction:column;
      justify-content:center;
      text-align:center;
  }}
  .mm-review-severity span {{
      color:{INK};
      font-size:0.9rem;
      font-weight:780;
  }}
  .mm-review-severity small {{
      color:{FAINT};
      font-size:0.61rem;
      text-transform:uppercase;
      letter-spacing:.07em;
  }}
  .mm-review-problem {{
      display:grid;
      grid-template-columns:140px minmax(0, 1fr);
      gap:12px;
      align-items:start;
      padding:11px 13px;
      margin-bottom:13px;
      border:1px solid #FDE68A;
      border-radius:8px;
      background:{WARNING_SOFT};
      color:#92400E;
      font-size:0.92rem;
      line-height:1.4;
  }}
  .mm-review-problem b {{
      font-size:0.64rem;
      text-transform:uppercase;
      letter-spacing:.07em;
      color:#B45309;
  }}
  .mm-review-subtitle {{
      color:{MUTED};
      font-size:0.86rem;
      font-weight:720;
      margin:2px 0 8px;
  }}
  .mm-review-action-grid {{
      display:grid;
      grid-template-columns:repeat(3, minmax(0, 1fr));
      gap:10px;
      margin:0 0 14px;
  }}
  .mm-review-action-grid div {{
      min-width:0;
      padding:11px 12px;
      border-radius:8px;
      border:1px solid rgba(79,70,229,.18);
      border-left:4px solid {ACCENT};
      background:
        linear-gradient(135deg, rgba(238,242,255,.9), rgba(255,255,255,.92)),
        linear-gradient(225deg, rgba(14,165,233,.08), transparent 45%);
      box-shadow:var(--shadow-sm);
  }}
  .mm-review-action-grid b {{
      display:block;
      color:{ACCENT_TEXT};
      font-size:0.88rem;
      font-weight:780;
      line-height:1.25;
      margin-bottom:5px;
  }}
  .mm-review-action-grid span {{
      display:block;
      color:{INK};
      font-size:0.82rem;
      line-height:1.35;
  }}
  .mm-review-action-grid small {{
      display:block;
      margin-top:7px;
      color:{MUTED};
      font-size:0.76rem;
      line-height:1.35;
  }}
  .st-key-review_editor_card {{
      padding:13px;
      margin-top:12px;
      border:1px solid {LINE};
      border-radius:8px !important;
      background:rgba(255,255,255,.78) !important;
      box-shadow:var(--shadow-sm) !important;
  }}
  .mm-review-trust-grid {{
      display:grid;
      grid-template-columns:repeat(2, minmax(0, 1fr));
      gap:8px;
      margin-bottom:12px;
  }}
  .mm-review-trust-grid div {{
      padding:10px;
      border:1px solid {LINE};
      border-radius:8px;
      background:{ELEVATED};
  }}
  .mm-review-trust-grid b {{
      display:block;
      color:{INK};
      font-size:1.08rem;
      line-height:1;
      font-weight:760;
      font-variant-numeric:tabular-nums;
  }}
  .mm-review-trust-grid span {{
      display:block;
      margin-top:7px;
      color:{MUTED};
      font-size:0.62rem;
      font-weight:760;
      text-transform:uppercase;
      letter-spacing:.07em;
  }}
  .mm-review-audit-row {{
      background:var(--bg);
      border:1px solid var(--bd);
      border-left:3px solid var(--fg);
      color:var(--fg);
      border-radius:0 8px 8px 0;
      padding:8px 10px;
      font-size:0.74rem;
      line-height:1.32;
      margin-bottom:6px;
  }}
  .mm-review-audit-row b {{ font-weight:760; }}
  .mm-review-audit-row span {{
      font-family:var(--mono);
      color:var(--fg);
      opacity:.88;
  }}
  .mm-review-audit-row small {{
      display:block;
      margin-top:3px;
      color:var(--fg);
      opacity:.68;
      font-size:0.65rem;
  }}
  @media (min-width: 1180px) {{
      .st-key-review_queue_panel,
      .st-key-review_context_panel {{
          position:sticky;
          top:86px;
          max-height:calc(100vh - 112px);
          overflow:auto;
      }}
  }}
  @media (max-width: 1180px) {{
      .mm-review-metrics {{
          grid-template-columns:repeat(3, minmax(0, 1fr));
      }}
      .mm-review-hero {{
          flex-direction:column;
      }}
  }}
  @media (max-width: 760px) {{
      .mm-review-metrics {{
          grid-template-columns:repeat(2, minmax(0, 1fr));
      }}
      .mm-review-case-head {{
          flex-direction:column;
      }}
      .mm-review-severity {{
          align-self:stretch;
          border-left:none;
          border-top:1px solid {LINE};
          padding:10px 0 0;
      }}
      .mm-review-problem {{
          grid-template-columns:1fr;
          gap:4px;
      }}
      .mm-review-action-grid {{
          grid-template-columns:1fr;
      }}
  }}

  .mm-req-hero {{
      display:flex;
      align-items:flex-start;
      justify-content:space-between;
      gap:18px;
      padding:20px;
      margin-bottom:12px;
      border:1px solid #BAE6FD;
      border-radius:8px;
      background:
        linear-gradient(135deg, rgba(14,165,233,.22), transparent 48%),
        linear-gradient(225deg, rgba(22,163,74,.15), transparent 42%),
        linear-gradient(180deg, #F8FDFF 0%, #F5FBF7 100%);
      box-shadow:var(--shadow-md);
      animation:mm-rise .34s cubic-bezier(.22,1,.36,1) both;
  }}
  .mm-req-hero-badge {{
      flex:0 0 auto;
      padding:7px 10px;
      border-radius:999px;
      border:1px solid #BAE6FD;
      background:#F0F9FF;
      color:#0369A1;
      font-size:0.65rem;
      font-weight:780;
      text-transform:uppercase;
      letter-spacing:.07em;
  }}
  .mm-req-metrics {{
      display:grid;
      grid-template-columns:repeat(6, minmax(0, 1fr));
      gap:10px;
      margin-bottom:12px;
  }}
  .mm-req-metrics div {{
      min-width:0;
      padding:13px 13px;
      border-radius:8px;
      border:1px solid {LINE};
      border-top:3px solid {ACCENT};
      background:
        linear-gradient(180deg, rgba(255,255,255,.98), rgba(248,250,252,.92));
      box-shadow:var(--shadow-sm);
  }}
  .mm-req-metrics div:nth-child(2) {{ border-top-color:{WARNING}; background:linear-gradient(180deg, #FFFBEB, rgba(255,255,255,.94)); }}
  .mm-req-metrics div:nth-child(3) {{ border-top-color:{SUCCESS}; background:linear-gradient(180deg, #F0FDF4, rgba(255,255,255,.94)); }}
  .mm-req-metrics div:nth-child(4) {{ border-top-color:#0EA5E9; background:linear-gradient(180deg, #F0F9FF, rgba(255,255,255,.94)); }}
  .mm-req-metrics div:nth-child(5) {{ border-top-color:#7C3AED; background:linear-gradient(180deg, #F5F3FF, rgba(255,255,255,.94)); }}
  .mm-req-metrics div:nth-child(6) {{ border-top-color:#DB2777; background:linear-gradient(180deg, #FDF2F8, rgba(255,255,255,.94)); }}
  .mm-req-metrics b {{
      display:block;
      color:{INK};
      font-size:1.35rem;
      line-height:1;
      font-weight:760;
      font-variant-numeric:tabular-nums;
  }}
  .mm-req-metrics span {{
      display:block;
      margin-top:7px;
      color:{MUTED};
      font-size:0.68rem;
      font-weight:760;
      text-transform:uppercase;
      letter-spacing:.07em;
      white-space:nowrap;
      overflow:hidden;
      text-overflow:ellipsis;
  }}
  .mm-req-metrics small {{
      display:block;
      margin-top:4px;
      color:{FAINT};
      font-size:0.76rem;
      line-height:1.25;
  }}
  .mm-req-stepbar {{
      display:grid;
      grid-template-columns:repeat(4, minmax(0, 1fr));
      gap:10px;
      margin:0 0 12px;
  }}
  .mm-req-stepbar div {{
      min-width:0;
      display:grid;
      grid-template-columns:34px minmax(0, 1fr);
      column-gap:10px;
      align-items:center;
      padding:10px 12px;
      border-radius:8px;
      border:1px solid rgba(14,165,233,.18);
      background:
        linear-gradient(135deg, rgba(240,249,255,.95), rgba(255,255,255,.92)),
        linear-gradient(225deg, rgba(240,253,244,.65), transparent 48%);
      box-shadow:var(--shadow-sm);
  }}
  .mm-req-stepbar b {{
      grid-row:1 / span 2;
      width:30px;
      height:30px;
      border-radius:8px;
      display:flex;
      align-items:center;
      justify-content:center;
      background:{INK};
      color:#fff;
      font-family:var(--mono);
      font-size:0.78rem;
  }}
  .mm-req-stepbar span {{
      color:{INK};
      font-size:0.88rem;
      font-weight:780;
      line-height:1.2;
  }}
  .mm-req-stepbar small {{
      color:{MUTED};
      font-size:0.74rem;
      line-height:1.3;
  }}
  .st-key-req_brief_panel,
  .st-key-req_results_panel,
  .st-key-req_strategy_panel {{
      background:{SURFACE} !important;
      border-radius:8px !important;
      box-shadow:var(--shadow-md) !important;
      animation:mm-rise .4s cubic-bezier(.22,1,.36,1) both;
  }}
  .st-key-req_brief_panel {{
      background:
        linear-gradient(180deg, rgba(255,255,255,.96), rgba(255,255,255,.88)),
        linear-gradient(135deg, rgba(14,165,233,.13), transparent 46%),
        linear-gradient(225deg, rgba(79,70,229,.08), transparent 42%) !important;
      font-size:0.96rem;
  }}
  .st-key-req_results_panel {{
      background:
        linear-gradient(180deg, rgba(255,255,255,.98), rgba(255,255,255,.92)),
        linear-gradient(135deg, rgba(22,163,74,.08), transparent 48%),
        linear-gradient(225deg, rgba(217,119,6,.075), transparent 40%) !important;
      font-size:0.96rem;
  }}
  .st-key-req_strategy_panel {{
      background:
        linear-gradient(180deg, rgba(255,255,255,.97), rgba(255,255,255,.88)),
        linear-gradient(135deg, rgba(22,163,74,.12), transparent 42%),
        linear-gradient(225deg, rgba(219,39,119,.07), transparent 44%) !important;
      font-size:0.96rem;
  }}
  .st-key-req_brief_panel textarea {{
      min-height:260px !important;
      font-size:0.96rem !important;
      line-height:1.55 !important;
  }}
  .st-key-req_brief_panel input,
  .st-key-req_results_panel input,
  .st-key-req_strategy_panel input,
  .st-key-req_brief_panel [data-baseweb="select"] *,
  .st-key-req_results_panel [data-baseweb="select"] *,
  .st-key-req_strategy_panel [data-baseweb="select"] * {{
      font-size:0.92rem !important;
  }}
  .mm-req-note {{
      margin-top:8px;
      padding:8px 10px;
      border:1px solid {LINE};
      border-radius:8px;
      background:{ELEVATED};
      color:{MUTED};
      font-size:0.82rem;
      line-height:1.45;
  }}
  .mm-req-chip-block {{
      padding:9px 0;
      border-bottom:1px solid {LINE};
  }}
  .mm-req-chip-block:last-child {{
      border-bottom:none;
  }}
  .mm-req-chip-block b {{
      display:block;
      color:{FAINT};
      font-size:0.62rem;
      font-weight:780;
      text-transform:uppercase;
      letter-spacing:.08em;
      margin-bottom:6px;
  }}
  .mm-req-fit-grid {{
      display:grid;
      grid-template-columns:repeat(2, minmax(0, 1fr));
      gap:8px;
      margin-bottom:12px;
  }}
  .mm-req-fit-grid div {{
      padding:10px;
      border:1px solid {LINE};
      border-radius:8px;
      background:{ELEVATED};
  }}
  .mm-req-fit-grid b {{
      display:block;
      color:{INK};
      font-size:1.22rem;
      line-height:1;
      font-weight:760;
      font-variant-numeric:tabular-nums;
  }}
  .mm-req-fit-grid span {{
      display:block;
      margin-top:7px;
      color:{MUTED};
      font-size:0.7rem;
      font-weight:760;
      text-transform:uppercase;
      letter-spacing:.07em;
  }}
  [class*="st-key-req_match_card_"] {{
      border-radius:8px !important;
      background:
        linear-gradient(180deg, rgba(255,255,255,.96), rgba(255,255,255,.9)),
        linear-gradient(135deg, rgba(79,70,229,.045), transparent 46%) !important;
      box-shadow:var(--shadow-sm) !important;
      margin-bottom:10px !important;
  }}
  .mm-req-card-head {{
      display:grid;
      grid-template-columns:58px 108px minmax(0, 1fr);
      align-items:center;
      gap:10px;
      margin-bottom:10px;
      padding-bottom:9px;
      border-bottom:1px solid {LINE};
  }}
  .mm-req-rank {{
      width:44px;
      height:44px;
      border-radius:8px;
      background:{INK};
      color:#fff;
      display:flex;
      align-items:center;
      justify-content:center;
      font-family:var(--mono);
      font-size:0.82rem;
      font-weight:760;
      box-shadow:0 6px 14px rgba(9,9,11,.16);
  }}
  .mm-req-score {{
      padding:8px 9px;
      border:1px solid #C7D2FE;
      border-radius:8px;
      background:{ACCENT_SOFT};
      text-align:center;
  }}
  .mm-req-score b {{
      display:block;
      color:{ACCENT_TEXT};
      font-size:1.05rem;
      line-height:1;
      font-weight:780;
      font-variant-numeric:tabular-nums;
  }}
  .mm-req-score span {{
      display:block;
      margin-top:5px;
      color:{ACCENT_TEXT};
      opacity:.74;
      font-size:0.58rem;
      text-transform:uppercase;
      letter-spacing:.07em;
      font-weight:760;
  }}
  .mm-req-components {{
      display:grid;
      grid-template-columns:repeat(3, minmax(0, 1fr));
      gap:7px;
      min-width:0;
  }}
  .mm-req-components div {{
      min-width:0;
      padding:7px 8px;
      border:1px solid {LINE};
      border-radius:8px;
      background:{SURFACE};
  }}
  .mm-req-components span {{
      display:block;
      color:{MUTED};
      font-size:0.6rem;
      font-weight:760;
      text-transform:uppercase;
      letter-spacing:.06em;
      white-space:nowrap;
      overflow:hidden;
      text-overflow:ellipsis;
  }}
  .mm-req-components b {{
      display:block;
      margin-top:5px;
      color:{INK};
      font-family:var(--mono);
      font-size:0.9rem;
      font-weight:650;
  }}
  [class*="st-key-req_match_card_"] .stButton>button {{
      min-height:38px;
      font-size:0.84rem;
      padding-left:0.55rem;
      padding-right:0.55rem;
  }}
  .st-key-req_add_ranked_slate button {{
      background:{ACCENT}; color:#fff; border-color:{ACCENT};
  }}
  .st-key-req_add_ranked_slate button:hover {{
      background:#4338CA; border-color:#4338CA; color:#fff;
  }}
  @media (min-width: 1180px) {{
      .st-key-req_brief_panel,
      .st-key-req_strategy_panel {{
          position:sticky;
          top:86px;
          max-height:calc(100vh - 112px);
          overflow:auto;
      }}
  }}
  @media (max-width: 1180px) {{
      .mm-req-metrics {{
          grid-template-columns:repeat(3, minmax(0, 1fr));
      }}
      .mm-req-hero {{
          flex-direction:column;
      }}
  }}
  @media (max-width: 760px) {{
      .mm-req-metrics,
      .mm-req-components,
      .mm-req-stepbar {{
          grid-template-columns:repeat(2, minmax(0, 1fr));
      }}
      .mm-req-card-head {{
          grid-template-columns:48px minmax(0, 1fr);
      }}
      .mm-req-components {{
          grid-column:1 / -1;
      }}
  }}

  .mm-sl-hero {{
      display:flex;
      align-items:flex-start;
      justify-content:space-between;
      gap:18px;
      padding:18px;
      margin-bottom:10px;
      border:1px solid {LINE};
      border-radius:8px;
      background:
        linear-gradient(135deg, rgba(79,70,229,.085), transparent 46%),
        linear-gradient(225deg, rgba(217,119,6,.055), transparent 38%),
        rgba(255,255,255,.94);
      box-shadow:var(--shadow-md);
      animation:mm-rise .34s cubic-bezier(.22,1,.36,1) both;
  }}
  .mm-sl-flow {{
      display:flex;
      align-items:center;
      gap:8px;
      flex:0 0 auto;
      padding-top:4px;
      color:{MUTED};
      font-size:0.68rem;
      font-weight:750;
      text-transform:uppercase;
      letter-spacing:.06em;
  }}
  .mm-sl-flow b {{
      display:block;
      width:34px;
      height:1px;
      background:{LINE_STRONG};
  }}
  .mm-sl-metrics {{
      display:grid;
      grid-template-columns:repeat(6, minmax(0, 1fr));
      gap:9px;
      margin-bottom:12px;
  }}
  .mm-sl-metrics div {{
      min-width:0;
      padding:11px 12px;
      border-radius:8px;
      border:1px solid {LINE};
      background:{SURFACE};
      box-shadow:var(--shadow-sm);
  }}
  .mm-sl-metrics b {{
      display:block;
      color:{INK};
      font-size:1.2rem;
      line-height:1;
      font-weight:760;
      font-variant-numeric:tabular-nums;
  }}
  .mm-sl-metrics span {{
      display:block;
      margin-top:7px;
      color:{MUTED};
      font-size:0.62rem;
      font-weight:760;
      text-transform:uppercase;
      letter-spacing:.07em;
      white-space:nowrap;
      overflow:hidden;
      text-overflow:ellipsis;
  }}
  .mm-sl-metrics small {{
      display:block;
      margin-top:4px;
      color:{FAINT};
      font-size:0.69rem;
      line-height:1.25;
  }}
  .st-key-sl_rail_panel,
  .st-key-sl_profile_panel,
  .st-key-sl_side_panel {{
      background:{SURFACE} !important;
      border-radius:8px !important;
      box-shadow:var(--shadow-md) !important;
      animation:mm-rise .4s cubic-bezier(.22,1,.36,1) both;
  }}
  .st-key-sl_profile_panel {{
      overflow:hidden;
      background:
        linear-gradient(180deg, rgba(255,255,255,.97), rgba(255,255,255,.9)),
        linear-gradient(135deg, rgba(79,70,229,.055), transparent 42%) !important;
  }}
  .st-key-sl_side_panel {{
      background:
        linear-gradient(180deg, rgba(255,255,255,.97), rgba(255,255,255,.9)),
        linear-gradient(135deg, rgba(22,163,74,.05), transparent 42%) !important;
  }}
  .mm-sl-pick {{
      display:flex;
      gap:10px;
      align-items:flex-start;
      background:{SURFACE};
      border:1px solid {LINE};
      border-left:3px solid {LINE_STRONG};
      border-radius:8px;
      padding:10px 11px;
      margin:8px 0 5px;
      min-height:78px;
      box-shadow:var(--shadow-sm);
      transition:border-color .15s ease, background .15s ease, transform .15s ease,
                 box-shadow .15s ease;
  }}
  .mm-sl-pick.warn {{ border-left-color:{WARNING}; }}
  .mm-sl-pick.ok {{ border-left-color:{SUCCESS}; }}
  .mm-sl-pick.accent {{ border-left-color:{ACCENT}; }}
  .mm-sl-pick.is-selected {{
      border-color:#C7D2FE;
      border-left-color:{ACCENT};
      background:{ACCENT_SOFT};
      box-shadow:0 0 0 3px rgba(79,70,229,.08), var(--shadow-sm);
      transform:translateY(-1px);
  }}
  .mm-sl-pick-body {{
      min-width:0;
      flex:1;
  }}
  .mm-sl-pick-top {{
      display:flex;
      align-items:center;
      justify-content:space-between;
      gap:8px;
      color:{INK};
      font-size:0.82rem;
      font-weight:760;
      line-height:1.25;
  }}
  .mm-sl-pick-top span {{
      white-space:nowrap;
      overflow:hidden;
      text-overflow:ellipsis;
  }}
  .mm-sl-pick-top b {{
      flex:0 0 auto;
      color:{MUTED};
      background:rgba(255,255,255,.72);
      border:1px solid {LINE};
      border-radius:999px;
      padding:2px 7px;
      font-size:0.58rem;
      text-transform:uppercase;
      letter-spacing:.06em;
  }}
  .mm-sl-pick-sub {{
      margin-top:5px;
      color:{MUTED};
      font-size:0.72rem;
      line-height:1.32;
      display:-webkit-box;
      -webkit-line-clamp:2;
      -webkit-box-orient:vertical;
      overflow:hidden;
  }}
  .mm-sl-pick-meta {{
      margin-top:7px;
      color:{FAINT};
      font-family:var(--mono);
      font-size:0.62rem;
      white-space:nowrap;
      overflow:hidden;
      text-overflow:ellipsis;
  }}
  .mm-sl-profile-head {{
      display:flex;
      gap:14px;
      align-items:flex-start;
      padding:14px 15px;
      margin-bottom:11px;
      border:1px solid {LINE};
      border-left:4px solid {ACCENT};
      border-radius:8px;
      background:{SURFACE};
      box-shadow:var(--shadow-sm);
  }}
  .mm-sl-profile-head.warn {{ border-left-color:{WARNING}; }}
  .mm-sl-profile-head.ok {{ border-left-color:{SUCCESS}; }}
  .mm-sl-profile-head.accent {{ border-left-color:{ACCENT}; }}
  .mm-sl-profile-title {{
      min-width:0;
      flex:1;
  }}
  .mm-sl-profile-title h2 {{
      margin:0;
      font-size:1.25rem;
      line-height:1.16;
      letter-spacing:0;
  }}
  .mm-sl-profile-title p {{
      margin:5px 0 8px;
      color:{MUTED};
      font-size:0.86rem;
      line-height:1.4;
  }}
  .mm-sl-profile-score {{
      align-self:stretch;
      min-width:98px;
      border-left:1px solid {LINE};
      padding-left:14px;
      display:flex;
      flex-direction:column;
      justify-content:center;
      text-align:center;
  }}
  .mm-sl-profile-score b {{
      color:{INK};
      font-size:1.05rem;
      line-height:1;
      font-weight:780;
  }}
  .mm-sl-profile-score span {{
      margin-top:5px;
      color:{FAINT};
      font-size:0.6rem;
      font-weight:760;
      text-transform:uppercase;
      letter-spacing:.07em;
  }}
  .mm-sl-basis {{
      padding:10px 12px;
      margin-bottom:12px;
      border:1px solid #C7D2FE;
      border-radius:8px;
      background:{ACCENT_SOFT};
      color:{ACCENT_TEXT};
      font-size:0.78rem;
      line-height:1.45;
  }}
  .st-key-sl_notes_panel,
  .st-key-sl_outreach_panel {{
      border-radius:8px !important;
      background:rgba(255,255,255,.78) !important;
      box-shadow:var(--shadow-sm) !important;
      margin-top:10px !important;
  }}
  .st-key-sl_notes_panel textarea,
  .st-key-sl_outreach_panel textarea {{
      font-size:0.82rem !important;
      line-height:1.45 !important;
  }}
  .mm-sl-side-grid {{
      display:grid;
      grid-template-columns:repeat(2, minmax(0, 1fr));
      gap:8px;
      margin-bottom:12px;
  }}
  .mm-sl-side-grid div {{
      padding:10px;
      border:1px solid {LINE};
      border-radius:8px;
      background:{ELEVATED};
  }}
  .mm-sl-side-grid b {{
      display:block;
      color:{INK};
      font-size:1.08rem;
      line-height:1;
      font-weight:760;
      font-variant-numeric:tabular-nums;
  }}
  .mm-sl-side-grid span {{
      display:block;
      margin-top:7px;
      color:{MUTED};
      font-size:0.62rem;
      font-weight:760;
      text-transform:uppercase;
      letter-spacing:.07em;
  }}
  .st-key-sl_rail_panel .stButton>button,
  .st-key-sl_profile_panel .stButton>button {{
      min-height:34px;
      font-size:0.76rem;
  }}
  .st-key-sl_open_profile button {{
      background:{ACCENT_SOFT}; color:{ACCENT_TEXT}; border-color:#C7D2FE;
  }}
  @media (min-width: 1180px) {{
      .st-key-sl_rail_panel,
      .st-key-sl_side_panel {{
          position:sticky;
          top:86px;
          max-height:calc(100vh - 112px);
          overflow:auto;
      }}
  }}
  @media (max-width: 1180px) {{
      .mm-sl-metrics {{
          grid-template-columns:repeat(3, minmax(0, 1fr));
      }}
      .mm-sl-hero {{
          flex-direction:column;
      }}
  }}
  @media (max-width: 760px) {{
      .mm-sl-metrics {{
          grid-template-columns:repeat(2, minmax(0, 1fr));
      }}
      .mm-sl-profile-head {{
          flex-direction:column;
      }}
      .mm-sl-profile-score {{
          align-self:stretch;
          border-left:none;
          border-top:1px solid {LINE};
          padding:10px 0 0;
      }}
  }}

  .mm-profile-hud {{ display:flex; gap:16px; align-items:flex-start; }}
  .mm-hud-avatar {{
      width:52px; height:52px; flex:0 0 52px; font-size:0.95rem; border-radius:8px;
      box-shadow:var(--shadow-sm);
  }}
  .mm-hud-name {{
      font-size:1.35rem; font-weight:700; letter-spacing:0; line-height:1.15; color:{INK};
  }}

  .mm-insight-h {{
      font-size:0.7rem; font-weight:700; letter-spacing:.08em;
      text-transform:uppercase; color:{MUTED};
  }}
  .mm-insight-stats {{ display:flex; gap:18px; margin-top:10px; }}
  .mm-insight-stats div {{ display:flex; flex-direction:column; }}
  .mm-insight-stats b {{
      font-size:1.35rem; letter-spacing:0; color:{INK}; line-height:1.1; font-weight:700;
  }}
  .mm-insight-stats span {{ font-size:0.7rem; color:{MUTED}; margin-top:3px; }}

  .mm-switch-head {{ font-weight:600; font-size:0.8rem; margin-bottom:4px; color:{MUTED}; }}
  .st-key-cand_switcher {{ background:{SURFACE} !important; }}
  .st-key-candidate_command {{
      overflow:hidden;
      background:
        linear-gradient(135deg, rgba(79,70,229,.09), transparent 42%),
        linear-gradient(225deg, rgba(15,118,110,.07), transparent 38%),
        {SURFACE} !important;
      border-radius:8px !important;
      box-shadow:var(--shadow-md) !important;
      margin-bottom:0.9rem !important;
  }}
  .st-key-candidate_command > div > div[data-testid="stVerticalBlock"] {{
      gap:0.75rem;
  }}
  .mm-profile-command-head {{
      display:flex;
      gap:16px;
      align-items:flex-start;
      min-width:0;
  }}
  .mm-profile-command-body {{
      min-width:0;
      flex:1 1 auto;
  }}
  .mm-profile-status {{
      display:inline-flex;
      align-items:center;
      gap:6px;
      padding:4px 9px;
      border-radius:999px;
      font-size:0.66rem;
      font-weight:750;
      text-transform:uppercase;
      letter-spacing:.06em;
      margin-bottom:9px;
  }}
  .mm-profile-status::before {{
      content:"";
      width:7px;
      height:7px;
      border-radius:50%;
  }}
  .mm-profile-status.ok {{
      background:{SUCCESS_SOFT};
      color:#15803D;
      border:1px solid #BBF7D0;
  }}
  .mm-profile-status.ok::before {{
      background:{SUCCESS};
      box-shadow:0 0 0 3px rgba(22,163,74,.16);
  }}
  .mm-profile-status.warn {{
      background:{WARNING_SOFT};
      color:#B45309;
      border:1px solid #FDE68A;
  }}
  .mm-profile-status.warn::before {{ background:{WARNING}; }}
  .mm-profile-role {{
      color:{MUTED};
      font-size:0.9rem;
      margin-top:4px;
      line-height:1.45;
  }}
  .mm-profile-source {{
      margin-top:5px;
      color:{FAINT};
      font-family:var(--mono);
      font-size:0.68rem;
      white-space:nowrap;
      overflow:hidden;
      text-overflow:ellipsis;
      max-width:100%;
  }}
  .mm-profile-labels {{
      margin-top:12px;
      max-width:100%;
  }}
  .mm-profile-metrics {{
      display:grid;
      grid-template-columns:repeat(2, minmax(0, 1fr));
      gap:7px;
      margin-top:12px;
  }}
  .mm-profile-metrics div {{
      min-height:58px;
      padding:10px 11px;
      border-radius:8px;
      border:1px solid {LINE};
      background:rgba(255,255,255,.82);
      box-shadow:var(--shadow-sm);
  }}
  .mm-profile-metrics b {{
      display:block;
      font-size:1.1rem;
      line-height:1;
      font-weight:750;
      color:{INK};
      font-variant-numeric:tabular-nums;
  }}
  .mm-profile-metrics span {{
      display:block;
      margin-top:7px;
      color:{MUTED};
      font-size:0.64rem;
      font-weight:750;
      text-transform:uppercase;
      letter-spacing:.07em;
  }}
  .mm-console-title {{
      font-size:0.68rem;
      font-weight:750;
      letter-spacing:.1em;
      text-transform:uppercase;
      color:{FAINT};
      margin:1px 0 8px;
  }}
  .mm-console-note {{
      color:{FAINT};
      font-size:0.72rem;
      font-family:var(--mono);
      margin:-2px 0 9px;
  }}
  .st-key-candidate_command .stSelectbox [data-baseweb="select"] > div {{
      min-height:36px;
  }}
  .st-key-candidate_command .stButton>button,
  .st-key-candidate_command [data-testid="stDownloadButton"] button {{
      min-height:34px;
      font-size:0.78rem;
  }}
  .st-key-candidate_command .st-key-cand_prev button,
  .st-key-candidate_command .st-key-cand_next button {{
      min-width:42px;
      padding:0.35rem 0.45rem;
  }}
  .st-key-candidate_command .st-key-cand_prev button p,
  .st-key-candidate_command .st-key-cand_next button p {{
      display:none;
  }}
  .st-key-candidate_command .st-key-dlc_json button {{
      background:#EEF2FF; color:#3730A3; border-color:#C7D2FE;
  }}
  .st-key-candidate_command .st-key-dlc_csv button {{
      background:{SUCCESS_SOFT}; color:#15803D; border-color:#BBF7D0;
  }}
  .st-key-candidate_command .st-key-dlc_pdf button {{
      background:{DANGER_SOFT}; color:#B91C1C; border-color:#FECACA;
  }}
  .st-key-candidate_command .st-key-dlc_docx button {{
      background:#EFF6FF; color:#1D4ED8; border-color:#BFDBFE;
  }}
  .st-key-profile_detail_panel,
  .st-key-profile_source_panel,
  .st-key-profile_trust_panel {{
      border-radius:8px !important;
      box-shadow:var(--shadow-md) !important;
      background:{SURFACE} !important;
  }}
  .st-key-profile_detail_panel {{
      background:
        linear-gradient(180deg, rgba(255,255,255,.96), rgba(255,255,255,.88)),
        linear-gradient(135deg, rgba(22,163,74,.06), transparent 36%) !important;
  }}
  .st-key-profile_source_panel {{
      overflow:hidden;
  }}
  .st-key-profile_trust_panel {{
      margin-top:0.85rem !important;
      background:
        linear-gradient(180deg, rgba(255,255,255,.96), rgba(255,255,255,.9)),
        linear-gradient(135deg, rgba(217,119,6,.07), transparent 42%) !important;
  }}
  .mm-profile-panel-title {{
      font-size:0.96rem;
      font-weight:750;
      color:{INK};
      letter-spacing:0;
      margin-bottom:2px;
  }}
  .mm-profile-panel-sub {{
      color:{MUTED};
      font-size:0.76rem;
      line-height:1.45;
      margin-bottom:12px;
  }}
  .mm-candidate-summary {{
      padding:11px 12px;
      margin:2px 0 14px;
      border-radius:8px;
      border:1px solid {LINE};
      background:{ELEVATED};
      color:{INK};
      font-size:0.86rem;
      line-height:1.55;
  }}
  .st-key-profile_detail_panel h3,
  .st-key-profile_detail_panel p,
  .st-key-profile_trust_panel p {{
      max-width:100%;
  }}
  .st-key-profile_source_panel iframe,
  .st-key-profile_source_panel embed,
  .st-key-profile_source_panel object {{
      border-radius:8px;
      border:1px solid {LINE};
      background:#fff;
  }}
  @media (min-width: 1180px) {{
      .st-key-profile_source_panel {{
          position:sticky;
          top:86px;
      }}
  }}
  @media (max-width: 980px) {{
      .mm-profile-metrics {{
          grid-template-columns:repeat(2, minmax(0, 1fr));
      }}
      .mm-profile-command-head {{
          flex-direction:column;
      }}
  }}

  .mm-flow {{ position:relative; padding:4px 0 2px 22px; margin:4px 0 8px; }}
  .mm-flow::before {{
      content:""; position:absolute; left:6px; top:10px; bottom:10px;
      width:2px; background:{LINE}; border-radius:2px;
  }}
  .mm-flow-item {{ position:relative; padding:0 0 18px 12px; }}
  .mm-flow-item:last-child {{ padding-bottom:2px; }}
  .mm-flow-item::before {{
      content:""; position:absolute; left:-19px; top:7px;
      width:10px; height:10px; border-radius:50%;
      background:{ACCENT}; border:2px solid {SURFACE};
      box-shadow:0 0 0 2px {ACCENT_SOFT};
  }}
  .mm-flow-item.is-intern::before {{ background:{MUTED}; box-shadow:none; }}
  .mm-flow-meta {{ color:{MUTED}; font-size:0.8rem; margin-top:2px; line-height:1.45; }}
  .mm-flow details {{ margin-top:6px; }}
  .mm-flow details summary {{
      cursor:pointer; color:{ACCENT_TEXT}; font-size:0.78rem; font-weight:650; list-style:none;
  }}
  .mm-flow details summary::-webkit-details-marker {{ display:none; }}
  .mm-flow details ul {{
      margin:6px 0 0; padding-left:18px; color:{INK}; font-size:0.82rem;
  }}

  .mm-section-break {{ display:flex; align-items:center; gap:10px; margin:28px 0 14px; }}
  .mm-section-break .tick {{
      width:14px; height:3px; border-radius:2px; background:var(--sc); flex-shrink:0;
  }}
  .mm-section-break .label {{
      font-size:0.68rem; font-weight:750; text-transform:uppercase;
      letter-spacing:.1em; color:var(--sc); white-space:nowrap;
  }}
  .mm-section-break .rule {{
      flex:1; height:1px;
      background:linear-gradient(to right, var(--sc)55, {LINE} 55%);
  }}

  .mm-glance-sec {{
      font-size:0.65rem; font-weight:700; letter-spacing:.1em;
      text-transform:uppercase; color:{FAINT}; margin:14px 0 6px;
  }}
  .mm-glance-sec:first-child {{ margin-top:2px; }}
  .mm-glance-row {{ display:flex; align-items:center; gap:8px; margin:5px 0; }}
  .mm-glance-row .lbl {{
      flex:0 0 110px; text-align:right; font-size:0.76rem; color:{INK};
      white-space:nowrap; overflow:hidden; text-overflow:ellipsis; font-weight:500;
  }}
  .mm-glance-row .track {{
      flex:1 1 auto; height:7px; border-radius:4px; background:{ELEVATED};
      overflow:hidden; border:1px solid {LINE};
  }}
  .mm-glance-row .fill {{ display:block; height:100%; border-radius:4px; background:{ACCENT}; }}
  .mm-glance-row .n {{
      flex:0 0 22px; font-size:0.72rem; color:{MUTED}; font-variant-numeric:tabular-nums;
  }}

  /* ========== WIDGETS ========== */
  .stButton>button {{
      border-radius:8px; border:1px solid {LINE}; font-weight:600;
      font-size:0.82rem; padding:0.35rem 0.85rem;
      background:{SURFACE}; color:{INK}; box-shadow:var(--shadow-sm);
      transition:background .15s ease, border-color .15s ease, box-shadow .15s ease, transform .12s ease;
  }}
  .stButton>button:hover {{
      border-color:{LINE_STRONG}; box-shadow:var(--shadow-md); transform:translateY(-1px);
  }}
  .stButton>button:active {{ transform:translateY(0); }}
  .stButton>button[kind="primary"] {{
      background:{INK}; border-color:{INK}; color:#fff; box-shadow:0 4px 12px rgba(9,9,11,.18);
  }}
  .stButton>button[kind="primary"]:hover {{
      background:#27272A; border-color:#27272A; color:#fff;
  }}

  .st-key-cand_prev button {{
      background:{SURFACE}; color:{INK}; border:1px solid {LINE}; font-weight:600;
  }}
  .st-key-cand_prev button:hover:enabled {{ border-color:{ACCENT}; color:{ACCENT_TEXT}; }}
  .st-key-cand_prev button:disabled {{ background:{ELEVATED}; color:{FAINT}; }}
  .st-key-cand_next button {{
      background:{INK}; color:#fff; border:1px solid {INK}; font-weight:600;
  }}
  .st-key-cand_next button:hover:enabled {{ background:#27272A; }}
  .st-key-cand_next button:disabled {{
      background:{ELEVATED}; color:{FAINT}; border-color:{LINE};
  }}
  .st-key-cand_remove button, .st-key-act_delete button {{
      background:{DANGER_SOFT}; color:#B91C1C; border:1px solid #FECACA; font-weight:600;
  }}
  .st-key-cand_remove button:hover, .st-key-act_delete button:hover {{
      background:{DANGER}; color:#fff; border-color:{DANGER};
  }}
  .st-key-cand_shortlist button {{ font-weight:600; }}
  .st-key-cand_shortlist button[kind="secondary"] {{
      background:{WARNING_SOFT}; color:#92400E; border:1px solid #FDE68A;
  }}
  .st-key-cand_go_sl button {{
      background:{ACCENT_SOFT}; color:{ACCENT_TEXT}; border:1px solid #C7D2FE; font-weight:600;
  }}
  .st-key-profile_exports [data-testid="stDownloadButton"] button {{ font-weight:600; }}
  .st-key-dl_json button {{ background:#EEF2FF; color:#3730A3; border-color:#C7D2FE; }}
  .st-key-dl_csv button {{ background:{SUCCESS_SOFT}; color:#15803D; border-color:#BBF7D0; }}
  .st-key-dl_pdf button {{ background:{DANGER_SOFT}; color:#B91C1C; border-color:#FECACA; }}
  .st-key-dl_docx button {{ background:#EFF6FF; color:#1D4ED8; border-color:#BFDBFE; }}

  /* Command bar */
  div[data-testid="stTextInput"]:has(input[aria-label="Search"]) > div {{
      border:1px solid {LINE} !important; border-radius:8px !important;
      background:{SURFACE} !important;
      box-shadow:var(--shadow-sm);
      min-height:48px;
      transition:border-color .15s ease, box-shadow .15s ease;
  }}
  div[data-testid="stTextInput"]:has(input[aria-label="Search"]) > div:focus-within {{
      border-color:{ACCENT} !important;
      box-shadow:0 0 0 4px {ACCENT_SOFT}, var(--shadow-md);
  }}
  .st-key-search_hero {{
      background:transparent !important;
      border:none !important;
      box-shadow:none !important;
      animation:mm-rise .38s cubic-bezier(.22,1,.36,1) both;
  }}
  .st-key-search_hero > div > div[data-testid="stVerticalBlock"] {{ gap:0.7rem; }}
  .st-key-search_mode_help {{ margin-top:-0.35rem !important; }}
  /* CCv2 hosts: let the custom chrome breathe */
  .st-key-search_hero [data-testid="stCustomComponentV2"],
  .st-key-results_panel [data-testid="stCustomComponentV2"] {{
      margin:0 !important;
  }}
  .st-key-search_hero div[data-testid="stButtonGroup"] label {{
      border-radius:999px !important; font-size:0.76rem !important;
      background:{ELEVATED}; border-color:{LINE}; font-weight:550;
      transition:border-color .18s ease, background .18s ease, transform .18s cubic-bezier(.22,1,.36,1),
                 box-shadow .18s ease;
  }}
  .st-key-search_hero div[data-testid="stButtonGroup"] label:hover {{
      border-color:{ACCENT}; background:{ACCENT_SOFT}; transform:translateY(-1px);
      box-shadow:0 2px 8px rgba(79,70,229,.12);
  }}
  .st-key-search_hero [data-testid="stPopoverButton"] {{
      color:{MUTED}; border-color:{LINE}; background:{SURFACE};
  }}
  .mm-cmd-hint {{
      display:inline-flex; align-items:center; gap:4px;
      font-family:var(--mono); font-size:0.65rem; color:{FAINT};
      background:{ELEVATED}; border:1px solid {LINE}; border-radius:6px;
      padding:2px 6px; margin-left:6px;
  }}
  .st-key-open_match_studio button {{
      background:{ACCENT_SOFT}; color:{ACCENT_TEXT}; border:1px solid #C7D2FE; font-weight:650;
  }}
  .st-key-open_match_studio button:hover {{
      background:#E0E7FF; border-color:{ACCENT};
  }}
  .st-key-open_import_studio button {{
      background:{SURFACE}; color:{INK}; border:1px solid {LINE}; font-weight:650;
  }}
  .st-key-jump_shortlist_from_search button {{
      background:{WARNING_SOFT}; color:#92400E; border:1px solid #FDE68A; font-weight:650;
  }}
  .st-key-resume_match {{
      background:linear-gradient(180deg, {ACCENT_SOFT} 0%, {SURFACE} 48px) !important;
      border:1px solid #C7D2FE !important; border-radius:8px !important;
      animation:mm-rise .32s cubic-bezier(.22,1,.36,1) both;
  }}
  .st-key-pool_import {{
      border:1px dashed {LINE_STRONG} !important; border-radius:8px !important;
      background:{ELEVATED} !important;
      animation:mm-rise .32s cubic-bezier(.22,1,.36,1) both;
  }}
  .st-key-pool_glance {{
      overflow:auto; min-height:160px;
      background:{SURFACE} !important;
      border-radius:8px !important;
      box-shadow:var(--shadow-md) !important;
      animation:mm-rise .45s cubic-bezier(.22,1,.36,1) .06s both;
  }}
  .st-key-results_panel {{
      background:{SURFACE} !important;
      box-shadow:var(--shadow-md) !important;
      border-radius:8px !important;
      margin-top:0.85rem !important;
      animation:mm-rise .42s cubic-bezier(.22,1,.36,1) .04s both;
  }}
  .st-key-results_panel > div > div[data-testid="stVerticalBlock"] {{ gap:0.65rem; }}
  .mm-results-bar {{
      display:flex; align-items:baseline; justify-content:space-between;
      gap:12px; margin:2px 0 4px; padding-bottom:8px;
      border-bottom:1px solid {LINE};
  }}
  .mm-results-bar-label {{
      font-size:0.72rem; font-weight:750; letter-spacing:.1em;
      text-transform:uppercase; color:{MUTED};
  }}
  .mm-results-bar-meta {{
      margin-left:10px; font-family:var(--mono); font-size:0.68rem;
      font-weight:500; letter-spacing:0; text-transform:none; color:{FAINT};
  }}
  .mm-filter-chip-row {{
      display:flex; flex-wrap:wrap; align-items:center; gap:2px;
      margin:2px 0 6px; animation:mm-fade .28s ease both;
  }}
  .mm-filter-chip {{
      display:inline-flex; align-items:center; gap:4px;
      background:{ACCENT_SOFT}; color:{ACCENT_TEXT}; border:1px solid #C7D2FE;
      border-radius:999px; padding:3px 10px; font-size:0.7rem; font-weight:650;
      margin:2px 4px 2px 0;
      transition:transform .15s ease, box-shadow .15s ease;
  }}
  .mm-filter-chip:hover {{
      transform:translateY(-1px); box-shadow:0 2px 6px rgba(79,70,229,.15);
  }}
  .mm-rail-sec {{
      margin:16px 0 8px;
      padding-top:12px;
      border-top:1px solid {LINE};
      font-size:0.65rem; font-weight:750; letter-spacing:.1em;
      text-transform:uppercase; color:{FAINT};
  }}
  .mm-rail-person {{
      padding:10px 11px;
      margin:7px 0 5px;
      border:1px solid {LINE};
      border-radius:8px;
      background:linear-gradient(180deg, #FFFFFF 0%, #FAFAFA 100%);
      box-shadow:var(--shadow-sm);
  }}
  .mm-rail-person .mm-name {{
      font-size:0.82rem;
      max-width:68%;
      white-space:nowrap;
      overflow:hidden;
      text-overflow:ellipsis;
  }}
  .mm-rail-pill {{
      display:inline-flex;
      align-items:center;
      margin-left:4px;
      padding:2px 6px;
      border-radius:999px;
      background:{ACCENT_SOFT};
      color:{ACCENT_TEXT};
      border:1px solid #C7D2FE;
      font-size:0.62rem;
      font-weight:700;
  }}
  .st-key-pool_glance .stButton>button {{
      min-height:32px;
      font-size:0.76rem;
      box-shadow:none;
  }}
  .st-key-filter_studio {{
      background:transparent !important; border-radius:0 !important;
      padding:2px 0 4px !important;
  }}
  .st-key-filter_studio [data-testid="stPopoverButton"] button {{
      background:{ELEVATED}; border:1px solid {LINE}; font-weight:650;
      transition:border-color .18s ease, background .18s ease, transform .15s ease, box-shadow .18s ease;
  }}
  .st-key-filter_studio [data-testid="stPopoverButton"] button:hover {{
      border-color:{ACCENT}; background:{ACCENT_SOFT}; color:{ACCENT_TEXT};
      transform:translateY(-1px); box-shadow:0 2px 8px rgba(79,70,229,.12);
  }}
  .st-key-results_panel [data-testid="stDataFrame"] {{
      animation:mm-fade .35s ease .08s both;
  }}
  .st-key-results_panel div[data-testid="stButtonGroup"] label {{
      transition:border-color .15s ease, background .15s ease, transform .15s ease;
  }}

  @media (min-width: 1180px) {{
      .st-key-pool_glance {{
          position:sticky;
          top:86px;
          max-height:calc(100vh - 112px);
          margin-top:0.85rem !important;
      }}
      .st-key-pool_glance .mm-kpi {{
          padding:11px 12px;
          margin-bottom:8px;
      }}
      .st-key-pool_glance .mm-kpi .v {{ font-size:1.2rem; }}
  }}

  @media (max-width: 1179px) {{
      .st-key-results_panel, .st-key-pool_glance {{
          width:100%;
          position:relative;
          max-height:none;
      }}
  }}

  @keyframes mm-rise {{
      from {{ opacity:0; transform:translateY(10px); }}
      to {{ opacity:1; transform:translateY(0); }}
  }}
  @keyframes mm-fade {{
      from {{ opacity:0; }}
      to {{ opacity:1; }}
  }}
  @media (prefers-reduced-motion: reduce) {{
      .st-key-search_hero, .st-key-results_panel, .st-key-pool_glance,
      .st-key-resume_match, .st-key-pool_import,
      .st-key-results_panel [data-testid="stDataFrame"],
      .mm-filter-chip-row {{
          animation:none !important;
      }}
  }}

  .st-key-cta_top div[data-testid="stButton"] button {{
      font-weight:700; animation:mm-cta-pulse 2.4s ease-in-out infinite;
  }}
  @keyframes mm-cta-pulse {{
      0%, 100% {{ box-shadow:0 4px 12px rgba(9,9,11,.15); }}
      50% {{ box-shadow:0 4px 20px rgba(79,70,229,.25), 0 0 0 4px {ACCENT_SOFT}; }}
  }}
  @media (prefers-reduced-motion: reduce) {{
      .st-key-cta_top div[data-testid="stButton"] button {{ animation:none; }}
  }}

  /* Chat */
  .st-key-chat_dock {{
      position:fixed; top:0; right:0; bottom:0;
      width:min(var(--mm-chat-w), 100vw); z-index:400;
      background:{SURFACE}; border:none; border-left:1px solid {LINE};
      border-radius:0 !important; margin-bottom:0 !important;
      box-shadow:-16px 0 48px rgba(9,9,11,.1); padding:14px 16px;
      display:flex; flex-direction:column;
  }}
  .st-key-chat_dock > div {{ display:flex; flex-direction:column; flex:1 1 auto; min-height:0; }}
  .st-key-chat_dock > div > div[data-testid="stVerticalBlock"] {{
      flex:1 1 auto; min-height:0; gap:0.5rem;
  }}
  .st-key-chat_dock .st-key-chat_messages {{
      flex:1 1 0 !important; min-height:0 !important; height:auto !important;
      max-height:none !important; overflow:hidden !important;
      border:none !important; border-radius:0 !important;
      margin-bottom:0 !important; background:transparent;
  }}
  .st-key-chat_dock .st-key-chat_messages > div {{
      height:100% !important; overflow-y:auto !important; padding-right:4px;
  }}
  .st-key-chat_clear_btn button, .st-key-chat_close_btn button {{
      border:none; background:transparent; color:{MUTED};
      padding:0.15rem 0.4rem; min-height:1.7rem;
  }}
  .st-key-chat_clear_btn button:hover, .st-key-chat_close_btn button:hover {{
      background:{ELEVATED}; color:{INK}; border:none;
  }}
  .mm-chat-title {{
      font-size:0.98rem; font-weight:700; letter-spacing:0; padding-top:2px;
  }}
  .mm-chat-live {{
      display:inline-flex; align-items:center; gap:5px;
      font-size:0.65rem; font-weight:700; color:{SUCCESS};
      text-transform:uppercase; letter-spacing:.06em; margin-left:8px;
  }}
  .mm-chat-live .dot {{
      width:6px; height:6px; border-radius:50%; background:{SUCCESS};
      box-shadow:0 0 0 3px rgba(22,163,74,.2);
  }}
  .st-key-chat_toggle_btn {{ position:fixed; top:12px; right:16px; z-index:390; width:auto; }}
  .st-key-chat_toggle_btn button {{
      background:{ACCENT}; color:#fff; border:none; border-radius:999px;
      padding:0.45rem 1.05rem; font-weight:650; font-size:0.84rem;
      box-shadow:0 6px 20px rgba(79,70,229,.28);
      transition:transform .15s ease, box-shadow .15s ease, background .15s ease;
  }}
  .st-key-chat_toggle_btn button:hover {{
      background:#4338CA; transform:translateY(-1px);
      box-shadow:0 8px 24px rgba(79,70,229,.35);
  }}
  .st-key-chat_dock [data-testid="stChatMessage"] {{ gap:8px; margin-bottom:2px; }}
  .st-key-chat_dock [data-testid="stChatMessage"]:has(
      [data-testid="stChatMessageContent"][aria-label="Chat message from user"]) {{
      flex-direction:row-reverse;
  }}
  .st-key-chat_dock [data-testid="stChatMessageContent"] {{
      border-radius:8px 8px 8px 3px; padding:9px 13px; max-width:84%;
      font-size:0.84rem; line-height:1.5;
  }}
  .st-key-chat_dock [data-testid="stChatMessageContent"][aria-label="Chat message from user"] {{
      background:{INK}; color:#fff; border-radius:8px 8px 3px 8px; margin-left:auto;
  }}
  .st-key-chat_dock [data-testid="stChatMessageContent"][aria-label="Chat message from user"] p {{
      color:#fff;
  }}
  .st-key-chat_dock [data-testid="stChatMessageContent"][aria-label="Chat message from assistant"] {{
      background:{ELEVATED}; border:1px solid {LINE};
  }}
  .st-key-chat_dock [data-testid="stChatInput"] {{ border-radius:999px !important; }}
  .st-key-chat_dock [data-testid="stChatInput"]:focus-within {{
      box-shadow:0 0 0 3px {ACCENT_SOFT};
  }}

  .st-key-profile_back_fab {{
      position:fixed; bottom:18px; right:24px; left:auto; transform:none; z-index:360; width:auto;
  }}
  .st-key-profile_back_fab button {{
      background:{INK}; color:#fff; border:1px solid {INK};
      border-radius:999px; padding:0.45rem 1.05rem; font-weight:650; font-size:0.82rem;
      box-shadow:var(--shadow-lg);
  }}

  .mm-loading {{
      display:flex; flex-direction:column; align-items:center;
      justify-content:center; padding:96px 20px; text-align:center;
  }}
  .mm-loading-ring {{
      width:48px; height:48px; border-radius:50%;
      border:2.5px solid {LINE}; border-top-color:{ACCENT};
      display:flex; align-items:center; justify-content:center;
      animation:mm-spin 0.85s linear infinite; margin-bottom:18px;
      box-shadow:0 0 0 8px {ACCENT_SOFT};
  }}
  .mm-loading-mark {{
      font-size:0.9rem; color:{ACCENT}; font-weight:800;
      animation:mm-spin-rev 0.85s linear infinite;
  }}
  .mm-loading-title {{
      font-size:1.25rem; font-weight:700; letter-spacing:0; color:{INK};
  }}
  .mm-loading-sub {{ font-size:0.85rem; color:{MUTED}; margin-top:6px; }}
  @keyframes mm-spin {{ to {{ transform:rotate(360deg); }} }}
  @keyframes mm-spin-rev {{ to {{ transform:rotate(-360deg); }} }}
  @media (prefers-reduced-motion: reduce) {{
      .mm-loading-ring, .mm-loading-mark {{ animation:none; }}
  }}

  div[data-testid="stVerticalBlockBorderWrapper"] {{
      border-radius:8px !important; margin-bottom:14px;
      border-color:{LINE} !important; background:{SURFACE};
      box-shadow:var(--shadow-sm);
  }}
  .stTabs [data-baseweb="tab-list"] {{ gap:2px; border-bottom:1px solid {LINE}; }}
  .stTabs [data-baseweb="tab"] {{
      font-size:0.82rem; padding:8px 14px; font-weight:550; color:{MUTED};
      border-radius:8px 8px 0 0;
  }}
  .stTabs [data-baseweb="tab"]:hover {{ color:{INK}; background:rgba(9,9,11,.03); }}
  .stTabs [aria-selected="true"] {{ color:{INK} !important; font-weight:700; }}
  .stTabs [data-baseweb="tab-highlight"] {{
      background:{ACCENT} !important; height:2px !important; border-radius:2px;
  }}
  .stTabs [data-baseweb="tab-border"] {{ background:transparent !important; }}
  div[data-testid="stExpander"] details {{
      border:1px solid {LINE}; border-radius:8px; background:{SURFACE};
      box-shadow:var(--shadow-sm);
  }}
  [data-testid="stDataFrame"] {{
      border:1px solid {LINE}; border-radius:8px; overflow:hidden;
      box-shadow:var(--shadow-sm);
  }}
  .stDataFrame {{ font-size:0.8rem; }}
  hr {{ margin:0.75rem 0; border-color:{LINE}; }}
  [data-testid="stMetric"] {{
      background:{SURFACE}; border:1px solid {LINE}; border-radius:8px;
      padding:10px 12px; box-shadow:var(--shadow-sm);
  }}
  div[data-testid="stMetricValue"] {{ font-size:1.25rem; font-weight:700; }}
  [data-testid="stTextInput"] input,
  [data-testid="stNumberInput"] input,
  [data-testid="stTextArea"] textarea {{
      font-family:var(--font) !important;
  }}
  .stSelectbox [data-baseweb="select"] > div,
  .stMultiSelect [data-baseweb="select"] > div {{
      background:{SURFACE} !important; border-color:{LINE} !important;
      border-radius:8px !important;
  }}

  ::-webkit-scrollbar {{ width:10px; height:10px; }}
  ::-webkit-scrollbar-thumb {{
      background:#D4D4D8; border-radius:8px; border:2px solid {CANVAS};
  }}
  ::-webkit-scrollbar-track {{ background:transparent; }}

  /* ========== DENSER PREMIUM SHELL OVERRIDES ========== */
  .block-container {{
      padding-top:0.35rem;
  }}
  section[data-testid="stSidebar"] {{
      background:
        linear-gradient(155deg, rgba(79,70,229,.075), transparent 38%),
        linear-gradient(24deg, rgba(14,165,233,.055), transparent 34%),
        linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 54%, #F1F5F9 100%) !important;
      border-right:1px solid #D9E0EA !important;
      box-shadow:10px 0 34px rgba(15,23,42,.06);
      backdrop-filter:blur(18px) saturate(1.12);
  }}
  section[data-testid="stSidebar"] .block-container {{
      padding:0.85rem 0.95rem 1rem;
  }}
  section[data-testid="stSidebar"] * {{ color:{INK}; }}
  section[data-testid="stSidebar"] .mm-brand-lockup {{
      padding:3px 6px 11px 6px;
      border-bottom-color:#DEE4EE !important;
  }}
  section[data-testid="stSidebar"] .mm-brand-name {{ color:{INK} !important; }}
  section[data-testid="stSidebar"] .mm-brand-sub {{ color:{MUTED} !important; }}
  section[data-testid="stSidebar"] .mm-mark {{
      background:linear-gradient(145deg, #4F46E5 0%, #0F766E 100%);
      box-shadow:0 8px 22px rgba(79,70,229,.23);
  }}
  section[data-testid="stSidebar"] .mm-rail-snapshot {{
      gap:6px; padding:8px 0 2px;
  }}
  section[data-testid="stSidebar"] .mm-rail-snapshot div {{
      padding:7px 7px;
      background:rgba(255,255,255,.82);
      border:1px solid #DEE4EE;
      box-shadow:0 1px 2px rgba(15,23,42,.04), inset 0 1px 0 rgba(255,255,255,.75);
  }}
  section[data-testid="stSidebar"] .mm-rail-snapshot b {{ color:{INK} !important; }}
  section[data-testid="stSidebar"] .mm-rail-snapshot span {{ color:{MUTED} !important; }}
  section[data-testid="stSidebar"] div[role="radiogroup"] {{ gap:2px; }}
  section[data-testid="stSidebar"] div[role="radiogroup"] label {{
      min-height:35px;
      padding:8px 9px 8px 12px;
      border-radius:8px;
  }}
  section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{
      background:rgba(255,255,255,.72);
      border-color:#D9E0EA;
      box-shadow:0 6px 16px rgba(15,23,42,.05);
  }}
  section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked),
  section[data-testid="stSidebar"] div[role="radiogroup"] label[data-selected="true"] {{
      background:linear-gradient(90deg, #EEF2FF 0%, rgba(255,255,255,.92) 100%);
      border-color:#C7D2FE;
      box-shadow:0 8px 20px rgba(79,70,229,.12), inset 0 1px 0 rgba(255,255,255,.9);
  }}
  section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked)::after,
  section[data-testid="stSidebar"] div[role="radiogroup"] label[data-selected="true"]::after {{
      background:{ACCENT};
      box-shadow:0 0 12px rgba(79,70,229,.3);
      top:6px; bottom:6px;
  }}
  section[data-testid="stSidebar"] div[role="radiogroup"] label p {{
      color:#52525B !important;
      font-size:0.81rem !important;
      padding-left:24px;
  }}
  section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p,
  section[data-testid="stSidebar"] div[role="radiogroup"] label[data-selected="true"] p {{
      color:{ACCENT_TEXT} !important;
  }}
  section[data-testid="stSidebar"] div[role="radiogroup"] label p::before {{
      color:#737B8C;
      font-size:17px;
  }}
  section[data-testid="stSidebar"] div[role="radiogroup"] label[data-selected="true"] p::before,
  section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p::before {{
      color:{ACCENT};
  }}
  section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-child(1),
  section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-child(5),
  section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-child(7) {{
      margin-top:23px;
  }}
  section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-child(1) {{
      margin-top:20px;
  }}
  section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-child(1)::before,
  section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-child(5)::before,
  section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-child(7)::before {{
      color:#7A8291;
      top:-17px;
      font-size:0.58rem;
  }}
  section[data-testid="stSidebar"] .mm-nav-purpose {{
      background:linear-gradient(180deg, rgba(255,255,255,.86), rgba(248,250,252,.78));
      border-color:#DEE4EE;
      color:{MUTED};
      margin:6px 0 3px;
      padding:8px 10px;
  }}
  section[data-testid="stSidebar"] div[data-testid="stExpander"] details {{
      background:rgba(255,255,255,.78) !important;
      border:1px solid #DEE4EE !important;
      box-shadow:0 1px 2px rgba(15,23,42,.04) !important;
  }}
  section[data-testid="stSidebar"] div[data-testid="stExpander"] summary {{
      min-height:34px;
  }}
  section[data-testid="stSidebar"] div[data-testid="stExpander"] summary p {{
      color:#3F3F46 !important;
  }}
  section[data-testid="stSidebar"] div[data-testid="stExpander"] svg {{
      color:#71717A !important;
  }}
  section[data-testid="stSidebar"] .stButton>button {{
      background:#FFFFFF;
      color:{INK};
      border-color:#D9E0EA;
      box-shadow:0 1px 2px rgba(15,23,42,.04);
  }}
  section[data-testid="stSidebar"] .stButton>button:hover {{
      background:#EEF2FF;
      color:{ACCENT_TEXT};
      border-color:#C7D2FE;
  }}
  section[data-testid="stSidebar"] .mm-rail-foot {{
      border-top:1px solid #DEE4EE;
      padding:12px 8px 2px;
      margin-top:6px;
  }}
  section[data-testid="stSidebar"] .mm-status-demo {{
      background:#FFF7ED;
      border-color:#FED7AA;
      color:#C2410C;
  }}
  section[data-testid="stSidebar"] .mm-rail-meta {{
      color:#7A8291;
  }}

  .st-key-app_chrome {{
      padding:4px 0 6px;
      margin:0 0 8px;
      background:rgba(248,250,252,.9);
      border-bottom:1px solid #DEE4EE;
  }}
  .mm-page-head {{ padding:0; }}
  .mm-page-kicker {{
      font-size:0.58rem;
      margin-bottom:0;
      letter-spacing:.09em;
  }}
  .mm-page-title {{
      font-size:1.02rem;
      line-height:1.1;
  }}
  .mm-page-sub {{
      font-size:0.71rem;
      margin-top:1px;
  }}
  .st-key-back_btn button, .st-key-home_btn button {{
      min-width:32px;
      height:30px;
      padding:0 9px;
  }}
  .st-key-kpi_band [data-testid="stPopoverButton"] {{
      min-height:30px;
      padding:4px 10px;
  }}
  .st-key-kpi_band [data-testid="stPopoverButton"] p {{
      font-size:0.69rem;
  }}
  .st-key-kpi_band [data-testid="stPopoverButton"] p strong {{
      font-size:0.72rem;
  }}

  .mm-review-hero,
  .mm-req-hero,
  .mm-sl-hero {{
      padding:12px 14px;
      margin-bottom:8px;
  }}
  .mm-review-flow,
  .mm-sl-flow {{
      gap:7px;
      margin:8px 0 0;
  }}
  .mm-review-metrics,
  .mm-req-metrics,
  .mm-sl-metrics,
  .mm-profile-metrics {{
      gap:7px;
      margin-top:8px;
      margin-bottom:8px;
  }}
  .mm-review-metrics div,
  .mm-req-metrics div,
  .mm-sl-metrics div,
  .mm-profile-metrics div {{
      padding:8px 10px;
  }}
  .mm-review-metrics b,
  .mm-req-metrics b,
  .mm-sl-metrics b,
  .mm-profile-metrics b {{
      font-size:1.08rem;
  }}
  .mm-review-metrics small,
  .mm-req-metrics small,
  .mm-sl-metrics small {{
      margin-top:4px;
  }}
  .st-key-candidate_command {{
      padding:12px 14px !important;
      margin-bottom:10px !important;
  }}
  .st-key-candidate_command > div > div[data-testid="stVerticalBlock"] {{
      gap:0.55rem;
  }}
  .mm-profile-command-head {{
      margin-bottom:8px;
  }}
  .mm-profile-status {{
      padding:4px 9px;
      font-size:0.66rem;
  }}
  .mm-profile-role {{
      font-size:1.04rem;
      margin-top:4px;
  }}
  .mm-profile-source {{
      margin-top:4px;
  }}

  .st-key-search_hero > div > div[data-testid="stVerticalBlock"] {{
      gap:0.45rem;
  }}
  .st-key-search_tools {{
      margin-top:0.25rem !important;
      margin-bottom:0.35rem !important;
  }}
  .st-key-search_tools [data-testid="stPopoverButton"],
  .st-key-search_tools .stButton>button {{
      min-height:34px;
  }}
  .st-key-results_panel {{
      margin-top:0.35rem !important;
  }}
  .st-key-pool_glance {{
      margin-top:0.35rem !important;
      min-height:132px;
  }}
  .st-key-results_panel > div > div[data-testid="stVerticalBlock"] {{
      gap:0.5rem;
  }}
  .mm-results-bar {{
      margin:0 0 2px;
      padding-bottom:6px;
  }}
  .mm-ai-panel-head {{
      display:flex;
      align-items:center;
      justify-content:space-between;
      gap:10px;
      margin:10px 0 8px;
      padding:8px 10px;
      border:1px solid #C7D2FE;
      border-radius:8px;
      background:linear-gradient(135deg, #EEF2FF 0%, #FFFFFF 72%);
      box-shadow:0 1px 2px rgba(79,70,229,.06);
  }}
  .mm-ai-panel-head span {{
      font-size:0.76rem;
      font-weight:760;
      letter-spacing:.08em;
      text-transform:uppercase;
      color:{ACCENT_TEXT};
  }}
  .mm-ai-panel-head b {{
      font-size:0.66rem;
      font-weight:700;
      color:{MUTED};
      background:#FFFFFF;
      border:1px solid #E0E7FF;
      border-radius:999px;
      padding:3px 8px;
      white-space:nowrap;
  }}
  .mm-ai-empty {{
      padding:14px 16px;
      border:1px dashed #C7D2FE;
      border-radius:8px;
      background:rgba(238,242,255,.56);
      color:{MUTED};
      font-size:0.82rem;
      line-height:1.5;
  }}
  .mm-llm-callout {{
      display:flex;
      align-items:center;
      gap:12px;
      margin:8px 0 12px;
      padding:10px 12px;
      border:1px solid #C7D2FE;
      border-left:3px solid {ACCENT};
      border-radius:8px;
      background:linear-gradient(135deg, rgba(238,242,255,.78), rgba(255,255,255,.94));
      box-shadow:0 1px 2px rgba(15,23,42,.04);
      color:{INK};
  }}
  .mm-llm-callout div {{
      min-width:0;
      flex:1 1 auto;
  }}
  .mm-llm-callout strong {{
      display:block;
      font-size:0.82rem;
      line-height:1.2;
      font-weight:760;
      color:{INK};
  }}
  .mm-llm-callout p {{
      margin:3px 0 0;
      color:{MUTED};
      font-size:0.74rem;
      line-height:1.35;
  }}
  .mm-llm-callout > b {{
      flex:0 0 auto;
      padding:3px 8px;
      border:1px solid #E0E7FF;
      border-radius:999px;
      background:#FFFFFF;
      color:{MUTED};
      font-size:0.62rem;
      font-weight:760;
      letter-spacing:.04em;
      text-transform:uppercase;
      white-space:nowrap;
  }}
  .mm-llm-chip {{
      flex:0 0 auto;
      display:inline-flex;
      align-items:center;
      justify-content:center;
      min-height:24px;
      padding:4px 9px;
      border-radius:999px;
      background:{ACCENT};
      color:#FFFFFF;
      font-size:0.64rem;
      font-weight:800;
      letter-spacing:.06em;
      text-transform:uppercase;
      white-space:nowrap;
  }}
  div[data-testid="stVerticalBlockBorderWrapper"] {{
      margin-bottom:10px;
  }}

  @media (max-width: 900px) {{
      .block-container {{ padding-left:0.85rem; padding-right:0.85rem; }}
      .st-key-kpi_band {{ flex-wrap:wrap; }}
      .mm-hud-name {{ font-size:1.15rem; }}
  }}
</style>
"""


def polish_fig(fig):
    """Charts sit on the canvas — no white plot paper."""
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK, size=11, family="Plus Jakarta Sans, system-ui, sans-serif"),
        title_font_color=INK,
        legend_font_color=MUTED,
    )
    fig.update_xaxes(
        color=MUTED, gridcolor="rgba(9,9,11,.06)",
        zerolinecolor="rgba(9,9,11,.08)", linecolor=LINE,
        tickfont=dict(color=MUTED), title_font=dict(color=MUTED),
    )
    fig.update_yaxes(
        color=MUTED, gridcolor="rgba(9,9,11,.06)",
        zerolinecolor="rgba(9,9,11,.08)", linecolor=LINE,
        tickfont=dict(color=MUTED), title_font=dict(color=MUTED),
    )
    return fig


def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


CHAT_OPEN_CSS = """
<style>
  section[data-testid="stMain"] .block-container {
      padding-right:calc(min(var(--mm-chat-w), 100vw) + 28px) !important;
      max-width:100% !important;
  }
  .st-key-profile_back_fab {
      left:auto;
      right:calc(min(var(--mm-chat-w), 100vw) + 24px);
      transform:none;
  }
</style>
"""


def inject_chat_open() -> None:
    st.markdown(CHAT_OPEN_CSS, unsafe_allow_html=True)


FLAG_CATEGORIES: dict[str, dict] = {
    "injection":  {"fg": "#B91C1C", "bg": "#FEF2F2", "icon": "⚠", "label": "Security",
                   "severity": 3},
    "pipeline":   {"fg": "#B91C1C", "bg": "#FEF2F2", "icon": "⛔", "label": "Pipeline error",
                   "severity": 3},
    "duplicate":  {"fg": "#6D28D9", "bg": "#F5F3FF", "icon": "⧉", "label": "Duplicate",
                   "severity": 2},
    "timeline":   {"fg": "#B45309", "bg": "#FFFBEB", "icon": "⏱", "label": "Timeline",
                   "severity": 2},
    "contact":    {"fg": "#B45309", "bg": "#FFFBEB", "icon": "✉", "label": "Contact",
                   "severity": 2},
    "abstained":  {"fg": "#B45309", "bg": "#FFFBEB", "icon": "⊘", "label": "Abstained",
                   "severity": 2},
    "quality":    {"fg": "#B45309", "bg": "#FFFBEB", "icon": "◐", "label": "Data quality",
                   "severity": 1},
    "llm_status": {"fg": "#71717A", "bg": "#F4F4F5", "icon": "⏳", "label": "LLM status",
                   "severity": 1},
    "repair":     {"fg": "#3730A3", "bg": "#EEF2FF", "icon": "🛠", "label": "Auto-repaired",
                   "severity": 0},
    "other":      {"fg": "#71717A", "bg": "#F4F4F5", "icon": "•", "label": "Note",
                   "severity": 1},
}

_FLAG_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("injection", ("injection", "white-on-white", "microscopic", "exfil", "hijack",
                  "prompt injection")),
    ("pipeline", ("pipeline degraded", "subagent(s) failed", "unresolved rule/llm conflict",
                 "pipeline stage failed")),
    ("duplicate", ("duplicate", "near-duplicate")),
    ("timeline", ("gap between", "overlap by", "end date precedes", "implausible tenure",
                 "starts before the earliest degree", "exceeds time since", "predates study")),
    ("contact", ("contact", "no usable contact", "malformed address", "not an email",
                "issn/isbn", "phone number")),
    ("llm_status", ("llm unavailable", "demo_mode", "no cached response", "replayed from cache")),
    ("abstained", ("abstained", "could not be located", "could not be derived",
                  "could not be confirmed", "discarded")),
    ("repair", ("repaired", "normalised", "re-joined", "un-wrapped", "stripped",
               "column layout")),
    ("quality", ("completeness", "evidence coverage", "extraction quality", "text layer",
                "low support", "no investment strategy", "no geography", "no sector",
                "core field", "poor text")),
]


def classify_flag(text: str) -> str:
    low = (text or "").lower()
    for category, keywords in _FLAG_RULES:
        if any(k in low for k in keywords):
            return category
    return "other"


def flag_card(text: str, prefix: str = "") -> str:
    cat = classify_flag(text)
    spec = FLAG_CATEGORIES[cat]
    body = html_escape((prefix + text) if prefix else text)
    return (
        f'<div style="background:{spec["bg"]};border:1px solid {spec["fg"]}33;'
        f'border-left:3px solid {spec["fg"]};color:{spec["fg"]};border-radius:0 8px 8px 0;'
        f'padding:8px 12px;font-size:0.8rem;margin-bottom:6px;line-height:1.5;'
        f'display:flex;gap:8px;align-items:flex-start">'
        f'<span style="flex-shrink:0">{spec["icon"]}</span>'
        f'<span><b style="font-size:0.65rem;text-transform:uppercase;letter-spacing:.05em;'
        f'opacity:.85">{spec["label"]}</b><br>{body}</span></div>')


def flag_list(texts: list[str], prefix: str = "") -> str:
    ordered = sorted(texts, key=lambda t: -FLAG_CATEGORIES[classify_flag(t)]["severity"])
    return "".join(flag_card(t, prefix) for t in ordered)


def html_escape(s: str) -> str:
    import html as _html
    return _html.escape(str(s))


METHOD_LABELS = {
    "llm": ("LLM", "Claude API"), "rule": ("𝑓", "rule / regex"),
    "hybrid": ("𝑓+LLM", "rule + LLM cross-check"), "derived": ("Σ", "computed in Python"),
    "human": ("✎", "human-corrected"),
}


def method_chip(method: str) -> str:
    icon, label = METHOD_LABELS.get(method, ("?", method))
    return (f'<span class="mm-chip mm-chip-plain" title="extraction method">'
            f'{icon} · {label}</span>')


def status_chip(status: str, label: str | None = None) -> str:
    fg, bg, _tip = STATUS.get(status, STATUS["missing"])
    return (f'<span class="mm-chip" style="background:{bg};color:{fg};'
            f'border-color:{fg}22">{label or status}</span>')


def chip(text: str, tone: str = "plain") -> str:
    if tone == "plain":
        return f'<span class="mm-chip mm-chip-plain">{text}</span>'
    fg, bg, _ = STATUS.get(tone, STATUS["missing"])
    return f'<span class="mm-chip" style="background:{bg};color:{fg}">{text}</span>'
