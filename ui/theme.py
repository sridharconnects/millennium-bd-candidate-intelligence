"""One restrained colour system, applied once.

Deliberately not the default Streamlit palette. This is a tool a recruiter would have
open for six hours a day, so it reads as an internal financial application: high
information density, muted surfaces, colour reserved for meaning rather than decoration.

Colour carries exactly four meanings and nothing else:
  verified (teal)   -- grounded in a source span
  abstained (amber) -- we saw a claim and could not prove it
  missing (grey)    -- the document never said
  conflict (red)    -- two extractors disagree
"""
from __future__ import annotations

import streamlit as st

INK = "#0F172A"
MUTED = "#64748B"
LINE = "#E2E8F0"
SURFACE = "#FFFFFF"
CANVAS = "#F8FAFC"
ACCENT = "#0F766E"
ACCENT_SOFT = "#CCFBF1"

STATUS = {
    "verified": ("#0F766E", "#CCFBF1", "grounded in a verified source span"),
    "derived": ("#1D4ED8", "#DBEAFE", "computed in Python from verified fields"),
    "abstained": ("#B45309", "#FEF3C7", "a value was proposed but its quote could not be verified — discarded"),
    "conflicted": ("#B91C1C", "#FEE2E2", "rule and model disagree — routed to human review"),
    "human_corrected": ("#6D28D9", "#EDE9FE", "corrected by a reviewer"),
    "missing": ("#64748B", "#F1F5F9", "not present in the document"),
    "unverified": ("#64748B", "#F1F5F9", "present but unverified"),
}

# Categorical series for charts. Muted, distinguishable, colour-blind safe ordering.
SERIES = ["#0F766E", "#1D4ED8", "#B45309", "#7E22CE", "#0E7490", "#BE123C",
          "#4D7C0F", "#A16207", "#475569", "#9333EA"]

CSS = f"""
<style>
  :root {{
    --ink:{INK}; --muted:{MUTED}; --line:{LINE};
    --surface:{SURFACE}; --canvas:{CANVAS}; --accent:{ACCENT};
  }}
  .stApp {{ background:{CANVAS}; }}

  /* Streamlit's own toolbar (hamburger menu, Deploy button) is fixed-position and
     reserves real vertical space; the previous 1.1rem block-container padding was
     less than that reservation, so our own title rendered partially underneath it
     and was clipped in every screenshot. Two ways to fix that: pad enough to clear
     it, or remove it. This is meant to read as a product, not a dev preview, so the
     toolbar is hidden outright and the container reclaims that space instead. */
  header[data-testid="stHeader"] {{ display:none; }}
  div[data-testid="stToolbar"] {{ display:none; }}
  .block-container {{ padding-top:1.6rem; padding-bottom:3rem; max-width:1480px; }}
  html, body, [class*="css"] {{
      font-family:"Inter",-apple-system,"Segoe UI",system-ui,sans-serif;
      color:{INK}; font-size:14px; }}
  h1,h2,h3,h4 {{ letter-spacing:-0.015em; font-weight:650; color:{INK}; }}
  h1 {{ font-size:1.45rem; }} h2 {{ font-size:1.15rem; }} h3 {{ font-size:1.0rem; }}

  /* dense sidebar */
  section[data-testid="stSidebar"] {{ background:{SURFACE}; border-right:1px solid {LINE}; }}
  section[data-testid="stSidebar"] .block-container {{ padding-top:1rem; }}

  /* Workspace nav: the most important navigation surface in the app, previously a
     bare radio list with no visual weight and no indication of what each page does.
     Each option becomes a real card -- padding, rounded corners, a hover state, and
     a left accent stripe plus tinted fill on whichever is selected -- so the current
     location is unmistakable at a glance rather than a small dot next to text. */
  section[data-testid="stSidebar"] div[role="radiogroup"] {{ gap:2px; }}
  section[data-testid="stSidebar"] div[role="radiogroup"] label {{
      padding:9px 10px 9px 12px; border-radius:8px; border-left:3px solid transparent;
      transition:background-color .12s ease, border-color .12s ease; cursor:pointer;
  }}
  section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{
      background:{CANVAS};
  }}
  section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {{
      background:{ACCENT_SOFT}; border-left-color:{ACCENT};
  }}
  section[data-testid="stSidebar"] div[role="radiogroup"] label p {{
      font-size:0.92rem !important; font-weight:550;
  }}
  section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {{
      color:{ACCENT}; font-weight:650;
  }}

  /* The one-line "what is this page for" callout under the nav -- updates the instant
     a page is picked, so a new user never has to guess from a one-word label alone. */
  .mm-nav-purpose {{
      font-size:0.78rem; line-height:1.45; color:{MUTED}; padding:2px 4px 0 14px;
      margin-top:-2px;
  }}

  .mm-card {{ background:{SURFACE}; border:1px solid {LINE}; border-radius:9px;
             padding:14px 16px; margin-bottom:10px; }}
  .mm-card:hover {{ border-color:#CBD5E1; }}
  .mm-row {{ display:flex; gap:10px; align-items:baseline; flex-wrap:wrap; }}
  .mm-name {{ font-weight:650; font-size:1.02rem; color:{INK}; }}
  .mm-sub {{ color:{MUTED}; font-size:0.83rem; }}
  .mm-mono {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:0.8rem; }}

  .mm-chip {{ display:inline-block; padding:1px 8px; border-radius:11px;
              font-size:0.72rem; font-weight:600; margin:2px 4px 2px 0;
              border:1px solid transparent; white-space:nowrap; }}
  .mm-chip-plain {{ background:{CANVAS}; color:{MUTED}; border-color:{LINE}; }}

  .mm-ev {{ background:#FFFBEB; border-left:3px solid #F59E0B; padding:7px 11px;
            font-size:0.82rem; border-radius:0 5px 5px 0; margin:5px 0; line-height:1.5; }}
  .mm-ev mark {{ background:#FDE68A; padding:1px 2px; border-radius:2px; font-weight:600; }}

  .mm-bar-wrap {{ background:#F1F5F9; border-radius:3px; height:9px; width:100%;
                  overflow:hidden; }}
  .mm-bar {{ height:9px; border-radius:3px; }}

  .mm-kpi {{ background:{SURFACE}; border:1px solid {LINE}; border-radius:9px;
             padding:11px 13px; min-width:96px; margin-bottom:18px; }}
  .mm-kpi .l {{ word-break:normal; overflow-wrap:normal; }}
  /* Nesting st.columns inside an already-narrow column (as the header briefly did)
     let a flex child shrink past its content's natural width, to the point where
     even whole-word wrapping ran out of room and the browser broke mid-word. The
     card's own min-width above is the floor that stops that regardless of whatever
     column layout it ends up nested inside later. */
  .mm-kpi .v {{ font-size:1.32rem; font-weight:680; letter-spacing:-0.02em;
                line-height:1.15; }}
  .mm-kpi .l {{ color:{MUTED}; font-size:0.72rem; text-transform:uppercase;
                letter-spacing:0.06em; margin-top:2px; }}
  .mm-kpi .h {{ color:{MUTED}; font-size:0.72rem; margin-top:5px; }}

  .mm-foot {{ position:sticky; bottom:0; background:rgba(248,250,252,.94);
              backdrop-filter:blur(6px); border-top:1px solid {LINE};
              padding:6px 2px; font-size:0.74rem; color:{MUTED};
              font-family:ui-monospace,Menlo,monospace; }}

  .mm-banner {{ background:{ACCENT_SOFT}; border:1px solid #99F6E4; color:#115E59;
                border-radius:8px; padding:9px 13px; font-size:0.82rem; margin-bottom:12px; }}
  .mm-warn {{ background:#FEF3C7; border:1px solid #FDE68A; color:#92400E;
              border-radius:8px; padding:9px 13px; font-size:0.82rem; margin-bottom:10px; }}
  .mm-danger {{ background:#FEE2E2; border:1px solid #FECACA; color:#991B1B;
                border-radius:8px; padding:9px 13px; font-size:0.82rem; margin-bottom:10px; }}
  .mm-synth {{ background:#EDE9FE; border:1px solid #DDD6FE; color:#5B21B6;
               border-radius:8px; padding:9px 13px; font-size:0.82rem; margin-bottom:10px;
               font-weight:600; }}

  /* Review queue record picker: a clickable card, not a hidden dropdown option. */
  .mm-review-pick {{ background:{SURFACE}; border:1.5px solid {LINE}; border-radius:8px;
                      padding:9px 11px; margin-bottom:4px; min-height:52px; }}
  .mm-review-pick.is-selected {{ border-color:{ACCENT}; background:{ACCENT_SOFT}; }}

  .stButton>button {{ border-radius:7px; border:1px solid {LINE}; font-weight:550;
                      font-size:0.84rem; padding:0.3rem 0.8rem; }}
  .stButton>button[kind="primary"] {{ background:{ACCENT}; border-color:{ACCENT}; }}

  /* The main search bar: an always-visible accent outline (not just on focus, so it
     reads as "live" the instant the page loads) plus a slow breathing glow -- a
     restrained stand-in for "this bar is actively listening", not a spinner, since
     nothing is actually in flight until a query is typed. Respects reduced motion. */
  div[data-testid="stTextInput"]:has(input[aria-label="Search"]) {{
      border-radius:10px;
  }}
  div[data-testid="stTextInput"]:has(input[aria-label="Search"]) > div {{
      border:1.5px solid {ACCENT} !important;
      border-radius:10px !important;
      box-shadow:0 0 0 3px {ACCENT_SOFT}, 0 1px 2px rgba(15,23,42,.04);
      animation:mm-search-breathe 2.8s ease-in-out infinite;
      transition:box-shadow .15s ease;
  }}
  div[data-testid="stTextInput"]:has(input[aria-label="Search"]) > div:focus-within {{
      animation:none;
      box-shadow:0 0 0 4px {ACCENT_SOFT};
  }}
  @keyframes mm-search-breathe {{
      0%, 100% {{ box-shadow:0 0 0 3px {ACCENT_SOFT}, 0 1px 2px rgba(15,23,42,.04); }}
      50% {{ box-shadow:0 0 0 6px {ACCENT_SOFT}, 0 1px 3px rgba(15,23,42,.06); }}
  }}
  @media (prefers-reduced-motion: reduce) {{
      div[data-testid="stTextInput"]:has(input[aria-label="Search"]) > div {{
          animation:none;
      }}
  }}
  /* Overview page's "Open the App" call-to-action -- the one button on the whole
     product that should visibly draw the eye, since its entire job is to move
     someone from the hero view into real functionality. Scoped via `.st-key-cta_top`,
     the class Streamlit derives from `st.button(..., key="cta_top")` -- confirmed
     against the live DOM rather than assumed, since an earlier version of this rule
     targeted `div[data-testid="element-container"]`, which this Streamlit version
     does not use (it exposes the same wrapper as a `.element-container` CLASS
     instead), so the rule silently matched nothing. Scoping matters here because
     every st.button in the app shares the same generic `kind="primary"` attribute --
     a plain attribute selector would blink Search's own Search button, Review's
     Save correction, Intake's Run pipeline, all of them at once, which is not the
     ask; one button should announce itself, not all of them. */
  .st-key-cta_top div[data-testid="stButton"] button {{
      animation:mm-cta-pulse 2.2s ease-in-out infinite;
      box-shadow:0 0 0 0 {ACCENT}66;
      font-weight:650;
  }}
  .st-key-cta_top div[data-testid="stButton"] button:hover {{
      animation:none; box-shadow:0 0 0 6px {ACCENT}22;
  }}
  @keyframes mm-cta-pulse {{
      0%   {{ box-shadow:0 0 0 0 {ACCENT}55; }}
      70%  {{ box-shadow:0 0 0 10px {ACCENT}00; }}
      100% {{ box-shadow:0 0 0 0 {ACCENT}00; }}
  }}
  @media (prefers-reduced-motion: reduce) {{
      .st-key-cta_top div[data-testid="stButton"] button {{ animation:none; }}
  }}

  /* Chat dock -- the full right-hand panel (app.py splits the page into
     main_col/chat_col via st.columns when `chat_open` is true; this styles the
     `st.container(key="chat_dock")` that wraps everything inside chat_col). Scoped
     via the same `.st-key-*` convention proven on the Overview CTA button.
     `position:sticky` (not `fixed`) pins it to the top of the viewport as the main
     column scrolls -- the same visual result as VS Code's always-present side panel --
     without `fixed`'s coordinate math or its risk of overlapping the header, since
     sticky still respects the column's own box and simply refuses to scroll past it. */
  .st-key-chat_dock {{
      background:{SURFACE}; border:1px solid {LINE}; border-radius:16px;
      padding:16px 18px 14px 18px; box-shadow:0 1px 3px rgba(15,23,42,.06),
      0 8px 24px rgba(15,23,42,.05); position:sticky; top:12px;
      max-height:calc(100vh - 24px); overflow-y:auto;
  }}
  .mm-chat-title {{ font-size:1.05rem; font-weight:750; letter-spacing:-0.01em;
                    padding-top:2px; }}
  .mm-chat-live {{ display:inline-flex; align-items:center; gap:5px;
                   font-size:0.68rem; font-weight:650; color:{ACCENT};
                   text-transform:uppercase; letter-spacing:.05em; margin-left:8px; }}
  .mm-chat-live .dot {{ width:6px; height:6px; border-radius:50%; background:{ACCENT};
                        box-shadow:0 0 0 0 {ACCENT}55; animation:mm-live-pulse 1.8s
                        ease-in-out infinite; }}
  @keyframes mm-live-pulse {{
      0%, 100% {{ box-shadow:0 0 0 0 {ACCENT}55; }}
      50% {{ box-shadow:0 0 0 4px {ACCENT}00; }}
  }}
  .st-key-chat_toggle_btn button {{ font-weight:600; }}

  /* Modern chat bubbles -- reskins Streamlit's own st.chat_message (verified live
     against this Streamlit version's DOM: stable `data-testid`s, not the fragile
     `.element-container` guess that broke the CTA button rule the first time
     around) rather than hand-rolling HTML bubbles, so Claude's markdown (bold,
     lists) still renders for free through `st.markdown`. */
  .st-key-chat_dock [data-testid="stChatMessage"] {{
      gap:8px; margin-bottom:2px; animation:mm-msg-in .18s ease-out;
  }}
  .st-key-chat_dock [data-testid="stChatMessage"]:has(
      [data-testid="stChatMessageContent"][aria-label="Chat message from user"]) {{
      flex-direction:row-reverse;
  }}
  .st-key-chat_dock [data-testid="stChatMessageContent"] {{
      border-radius:16px 16px 16px 4px; padding:9px 14px; max-width:84%;
      font-size:0.87rem; line-height:1.5;
  }}
  .st-key-chat_dock [data-testid="stChatMessageContent"][aria-label="Chat message from user"] {{
      background:{ACCENT}; color:#ffffff; border-radius:16px 16px 4px 16px;
      margin-left:auto;
  }}
  .st-key-chat_dock [data-testid="stChatMessageContent"][aria-label="Chat message from user"] p {{
      color:#ffffff;
  }}
  .st-key-chat_dock [data-testid="stChatMessageContent"][aria-label="Chat message from assistant"] {{
      background:{CANVAS}; border:1px solid {LINE};
  }}
  .st-key-chat_dock [data-testid^="stChatMessageAvatar"] {{
      width:26px; height:26px; border-radius:50%; display:flex;
      align-items:center; justify-content:center;
  }}
  .st-key-chat_dock [data-testid="stChatMessageAvatarUser"] {{
      background:{INK}; color:#fff;
  }}
  .st-key-chat_dock [data-testid="stChatMessageAvatarAssistant"] {{
      background:{ACCENT_SOFT}; color:{ACCENT};
  }}
  @keyframes mm-msg-in {{
      from {{ opacity:0; transform:translateY(4px); }}
      to   {{ opacity:1; transform:translateY(0); }}
  }}
  @media (prefers-reduced-motion: reduce) {{
      .st-key-chat_dock [data-testid="stChatMessage"] {{ animation:none; }}
      .mm-chat-live .dot {{ animation:none; }}
  }}

  /* Modern pill input + suggestion chips inside the dock. */
  .st-key-chat_dock [data-testid="stChatInput"] {{
      border-radius:22px !important; box-shadow:0 1px 2px rgba(15,23,42,.04);
  }}
  .st-key-chat_dock [data-testid="stChatInput"]:focus-within {{
      box-shadow:0 0 0 3px {ACCENT_SOFT};
  }}
  .st-key-chat_dock div[data-testid="stButtonGroup"] label {{
      border-radius:14px !important; font-size:0.8rem !important;
      transition:transform .1s ease, box-shadow .1s ease;
  }}
  .st-key-chat_dock div[data-testid="stButtonGroup"] label:hover {{
      transform:translateY(-1px); box-shadow:0 2px 6px rgba(15,23,42,.08);
  }}

  /* Labelled, coloured section dividers (components.section_break) -- used on pages
     long enough that a bare `st.divider()` repeated N times stops telling the reader
     anything about where they are. Colour cycles through the same categorical
     palette the charts use, so each section reads as visually distinct without
     introducing a second colour system. */
  .mm-section-break {{ display:flex; align-items:center; gap:10px; margin:26px 0 14px 0; }}
  .mm-section-break .tick {{ width:16px; height:3px; border-radius:2px;
                             background:var(--sc); flex-shrink:0; }}
  .mm-section-break .label {{ font-size:0.72rem; font-weight:750; text-transform:uppercase;
                              letter-spacing:.08em; color:var(--sc); white-space:nowrap; }}
  .mm-section-break .rule {{ flex:1; height:1px;
                             background:linear-gradient(to right, var(--sc)66, {LINE} 60%); }}

  /* Branded cold-start loading screen (components.loading_screen). Counter-rotating
     the mark against the ring keeps the ◧ glyph roughly upright while it spins --
     a small touch that reads as designed rather than as a borrowed spinner. */
  .mm-loading {{ display:flex; flex-direction:column; align-items:center;
                 justify-content:center; padding:90px 20px 100px 20px;
                 text-align:center; }}
  .mm-loading-ring {{ width:60px; height:60px; border-radius:50%;
                      border:3px solid {LINE}; border-top-color:{ACCENT};
                      display:flex; align-items:center; justify-content:center;
                      animation:mm-spin 1.05s linear infinite; margin-bottom:20px;
                      box-shadow:0 0 0 8px {ACCENT_SOFT}; }}
  .mm-loading-mark {{ font-size:1.5rem; color:{ACCENT};
                      animation:mm-spin-rev 1.05s linear infinite; }}
  .mm-loading-title {{ font-size:1.55rem; font-weight:760; letter-spacing:-0.02em;
                       color:{INK}; animation:mm-fade-up .45s ease-out; }}
  .mm-loading-sub {{ font-size:0.9rem; color:{MUTED}; margin-top:4px;
                     animation:mm-fade-up .45s ease-out .08s both; }}
  @keyframes mm-spin {{ to {{ transform:rotate(360deg); }} }}
  @keyframes mm-spin-rev {{ to {{ transform:rotate(-360deg); }} }}
  @keyframes mm-fade-up {{
      from {{ opacity:0; transform:translateY(5px); }}
      to   {{ opacity:1; transform:translateY(0); }}
  }}
  @media (prefers-reduced-motion: reduce) {{
      .mm-loading-ring, .mm-loading-mark {{ animation:none; }}
  }}

  /* `st.container(border=True)` -- used to give major page sections a visible
     boundary and consistent breathing room, replacing sections that used to butt
     directly against each other with nothing but a thin default hairline (or
     nothing at all) between them. Testid confirmed against this Streamlit
     version's live DOM rather than assumed. */
  div[data-testid="stVerticalBlockBorderWrapper"] {{
      border-radius:12px !important; margin-bottom:16px;
  }}
  div[data-testid="stVerticalBlockBorderWrapper"] > div {{ border-radius:12px; }}

  div[data-testid="stMetricValue"] {{ font-size:1.3rem; }}
  .stTabs [data-baseweb="tab"] {{ font-size:0.86rem; padding:6px 13px; }}
  div[data-testid="stExpander"] details {{ border:1px solid {LINE}; border-radius:8px;
                                           background:{SURFACE}; }}
  .stDataFrame {{ font-size:0.82rem; }}
  hr {{ margin:0.8rem 0; border-color:{LINE}; }}
</style>
"""


def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------- flag classification
# The pipeline emits dozens of distinct warning/flag/reason strings (repairs, gaps,
# duplicates, injection hits, abstentions, degraded subagents, ...). Left as a flat
# bullet list they are unscannable -- a recruiter cannot tell "the CV has a resume gap"
# apart from "the pipeline crashed on this document" apart from "we just repaired a
# ligature, this is informational." Every flag is classified once, here, into a small
# fixed set of categories with a distinct colour and icon, and every page that renders
# flags (Review, Intake, Candidate/Lineage) goes through the same classifier -- so a red
# card means the same thing everywhere in the app.
FLAG_CATEGORIES: dict[str, dict] = {
    "injection":  {"fg": "#B91C1C", "bg": "#FEE2E2", "icon": "⚠", "label": "Security",
                   "severity": 3},
    "pipeline":   {"fg": "#B91C1C", "bg": "#FEE2E2", "icon": "⛔", "label": "Pipeline error",
                   "severity": 3},
    "duplicate":  {"fg": "#7E22CE", "bg": "#EDE9FE", "icon": "⧉", "label": "Duplicate",
                   "severity": 2},
    "timeline":   {"fg": "#B45309", "bg": "#FEF3C7", "icon": "⏱", "label": "Timeline",
                   "severity": 2},
    "contact":    {"fg": "#B45309", "bg": "#FEF3C7", "icon": "✉", "label": "Contact",
                   "severity": 2},
    "abstained":  {"fg": "#B45309", "bg": "#FEF3C7", "icon": "⊘", "label": "Abstained",
                   "severity": 2},
    "quality":    {"fg": "#B45309", "bg": "#FEF3C7", "icon": "◐", "label": "Data quality",
                   "severity": 1},
    "llm_status": {"fg": "#475569", "bg": "#E2E8F0", "icon": "⏳", "label": "LLM status",
                   "severity": 1},
    "repair":     {"fg": "#1D4ED8", "bg": "#DBEAFE", "icon": "🛠", "label": "Auto-repaired",
                   "severity": 0},
    "other":      {"fg": "#64748B", "bg": "#F1F5F9", "icon": "•", "label": "Note",
                   "severity": 1},
}

# Ordered (specific -> general); the first substring match wins. Text is matched
# case-insensitively against the exact phrasing the pipeline itself generates (see
# ingest.py, sanitize.py, validate.py, orchestrator.py, agents/*.py).
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
    """One flag/warning/reason, colour-coded by what kind of thing it actually is."""
    cat = classify_flag(text)
    spec = FLAG_CATEGORIES[cat]
    body = html_escape((prefix + text) if prefix else text)
    return (
        f'<div style="background:{spec["bg"]};border:1px solid {spec["fg"]}33;'
        f'border-left:3px solid {spec["fg"]};color:{spec["fg"]};border-radius:0 7px 7px 0;'
        f'padding:7px 11px;font-size:0.82rem;margin-bottom:6px;line-height:1.5;'
        f'display:flex;gap:8px;align-items:flex-start">'
        f'<span style="flex-shrink:0">{spec["icon"]}</span>'
        f'<span><b style="font-size:0.68rem;text-transform:uppercase;letter-spacing:.04em;'
        f'opacity:.8">{spec["label"]}</b><br>{body}</span></div>')


def flag_list(texts: list[str], prefix: str = "") -> str:
    """A block of flags, most severe first, each colour-coded by category."""
    ordered = sorted(texts, key=lambda t: -FLAG_CATEGORIES[classify_flag(t)]["severity"])
    return "".join(flag_card(t, prefix) for t in ordered)


def html_escape(s: str) -> str:
    import html as _html
    return _html.escape(str(s))


# How a field's value was actually produced. Shown next to every field so "how is this
# happening via LLM" has a direct, per-field answer rather than a one-line disclaimer
# somewhere else on the page.
METHOD_LABELS = {
    "llm": ("🤖", "LLM (Claude API)"), "rule": ("𝑓", "rule / regex"),
    "hybrid": ("𝑓+🤖", "rule + LLM cross-check"), "derived": ("Σ", "computed in Python"),
    "human": ("✎", "human-corrected"),
}


def method_chip(method: str) -> str:
    icon, label = METHOD_LABELS.get(method, ("?", method))
    return (f'<span class="mm-chip mm-chip-plain" title="extraction method">'
            f'{icon} {label}</span>')


def status_chip(status: str, label: str | None = None) -> str:
    fg, bg, _tip = STATUS.get(status, STATUS["missing"])
    return (f'<span class="mm-chip" style="background:{bg};color:{fg};'
            f'border-color:{fg}22">{label or status}</span>')


def chip(text: str, tone: str = "plain") -> str:
    if tone == "plain":
        return f'<span class="mm-chip mm-chip-plain">{text}</span>'
    fg, bg, _ = STATUS.get(tone, STATUS["missing"])
    return f'<span class="mm-chip" style="background:{bg};color:{fg}">{text}</span>'
