"""Premium Search surfaces via Custom Components v2 (inline).

Streamlit widgets alone always look like Streamlit. These components own the
command bar and dense result list so the desk reads as a product, not a form.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import streamlit as st

_COMMAND_HTML = """
<div class="mm-cmd" id="root">
  <div class="mm-cmd-glow"></div>
  <div class="mm-cmd-inner">
    <div class="mm-cmd-head">
      <div>
        <div class="mm-cmd-kicker">Workspace · Search</div>
        <div class="mm-cmd-title">Find candidates</div>
      </div>
      <div class="mm-cmd-hint"><span class="mm-cmd-dot"></span> NLP search</div>
    </div>
    <div class="mm-cmd-field">
      <span class="mm-cmd-ico" aria-hidden="true">⌕</span>
      <input id="q" type="text" autocomplete="off" spellcheck="false"
             placeholder="Describe who you need — healthcare L/S in APAC, no banking…" />
      <button id="go" type="button">Run</button>
    </div>
    <div class="mm-cmd-row">
      <div class="mm-seg" id="modes" role="tablist"></div>
      <select id="show" aria-label="Show"></select>
      <div class="mm-cmd-spacer"></div>
      <div class="mm-cmd-meta" id="meta"></div>
    </div>
    <div class="mm-ex" id="examples"></div>
  </div>
</div>
"""

_COMMAND_CSS = """
:host, .mm-cmd {
  --ink: #09090B; --muted: #71717A; --faint: #A1A1AA; --line: #E4E4E7;
  --surface: #FFFFFF; --elev: #FAFAFA; --accent: #4F46E5; --accent-soft: #EEF2FF;
  --accent-text: #3730A3; --teal: #0F766E; --sky: #0284C7;
  --shadow: 0 18px 50px rgba(9,9,11,.10), 0 2px 8px rgba(9,9,11,.04);
  font-family: "Plus Jakarta Sans", -apple-system, "Segoe UI", system-ui, sans-serif;
  color: var(--ink); box-sizing: border-box;
}
*, *::before, *::after { box-sizing: border-box; }
.mm-cmd {
  position: relative; border-radius: 8px; overflow: hidden;
  background:
    linear-gradient(135deg, rgba(79,70,229,.08), transparent 44%),
    linear-gradient(180deg, #fff 0%, #FAFAFA 100%);
  border: 1px solid var(--line); box-shadow: var(--shadow);
  animation: rise .42s cubic-bezier(.22,1,.36,1) both;
}
.mm-cmd-glow {
  position: absolute; inset: -40% -20% auto -20%; height: 140px; pointer-events: none;
  background: linear-gradient(90deg, rgba(79,70,229,.16), rgba(14,165,233,.10), transparent);
}
.mm-cmd-inner { position: relative; padding: 12px 14px 10px; }
.mm-cmd-head {
  display: flex; align-items: center; justify-content: space-between;
  gap: 10px; margin-bottom: 8px;
}
.mm-cmd-kicker {
  font-size: 0.62rem; font-weight: 750; letter-spacing: .12em;
  text-transform: uppercase; color: var(--faint); margin-bottom: 2px;
}
.mm-cmd-title {
  font-size: 1.08rem; font-weight: 700; letter-spacing: 0; line-height: 1.15;
}
.mm-cmd-hint {
  display: inline-flex; align-items: center; gap: 6px;
  font-size: 0.68rem; color: var(--muted); white-space: nowrap;
  background: #fff; border: 1px solid var(--line); border-radius: 999px;
  padding: 3px 8px;
}
.mm-cmd-dot {
  width: 7px; height: 7px; border-radius: 50%;
  background: var(--teal); box-shadow: 0 0 0 3px rgba(15,118,110,.12);
}
.mm-cmd-field {
  display: flex; align-items: center; gap: 8px;
  background: #fff; border: 1.5px solid #D4D4D8; border-radius: 8px;
  padding: 3px 3px 3px 12px; min-height: 46px;
  box-shadow: inset 0 1px 0 rgba(255,255,255,.8), 0 1px 2px rgba(9,9,11,.03);
  transition: border-color .18s ease, box-shadow .18s ease, transform .18s ease;
}
.mm-cmd-field:focus-within {
  border-color: var(--accent);
  box-shadow: 0 0 0 4px var(--accent-soft), 0 8px 24px rgba(79,70,229,.12);
  transform: translateY(-1px);
}
.mm-cmd-ico { color: var(--faint); font-size: 1.15rem; flex: 0 0 auto; }
#q {
  flex: 1 1 auto; border: 0; outline: none; background: transparent;
  font: inherit; font-size: 0.9rem; letter-spacing: 0; color: var(--ink);
  min-width: 0;
}
#q::placeholder { color: #A1A1AA; }
#go {
  flex: 0 0 auto; border: 0; border-radius: 8px; cursor: pointer;
  background: linear-gradient(135deg, var(--ink), #27272A); color: #fff; font: inherit; font-weight: 650;
  font-size: 0.8rem; padding: 0 13px; height: 36px;
  transition: background .15s ease, transform .12s ease, box-shadow .15s ease;
  box-shadow: 0 4px 14px rgba(9,9,11,.2);
}
#go:hover { background: #27272A; transform: translateY(-1px); }
#go:active { transform: translateY(0); }
.mm-cmd-row {
  display: flex; align-items: center; gap: 8px; margin-top: 8px; flex-wrap: wrap;
}
.mm-seg { display: none; gap: 2px; padding: 3px; background: var(--elev);
  border: 1px solid var(--line); border-radius: 999px; }
.mm-seg button {
  border: 0; background: transparent; color: var(--muted); cursor: pointer;
  font: inherit; font-size: 0.7rem; font-weight: 600; padding: 4px 10px;
  border-radius: 999px; transition: all .15s ease;
}
.mm-seg button[aria-selected="true"] {
  background: #fff; color: var(--ink); box-shadow: 0 1px 3px rgba(9,9,11,.08);
}
.mm-seg button:hover { color: var(--ink); }
#show {
  border: 1px solid var(--line); background: #fff; border-radius: 999px;
  font: inherit; font-size: 0.7rem; font-weight: 600; color: var(--ink);
  padding: 5px 10px; cursor: pointer; outline: none;
}
#show:focus { border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-soft); }
.mm-cmd-spacer { flex: 1 1 auto; }
.mm-cmd-meta {
  font-family: "JetBrains Mono", ui-monospace, Menlo, monospace;
  font-size: 0.68rem; color: var(--faint);
}
.mm-ex { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 8px; }
.mm-ex button {
  border: 1px solid var(--line); background: var(--elev); color: var(--ink);
  border-radius: 999px; font: inherit; font-size: 0.7rem; font-weight: 550;
  padding: 4px 10px; cursor: pointer;
  transition: border-color .15s ease, background .15s ease, transform .12s ease;
}
.mm-ex button:hover {
  border-color: #C7D2FE; background: var(--accent-soft); color: var(--accent-text);
  transform: translateY(-1px);
}
@keyframes rise {
  from { opacity: 0; transform: translateY(12px); }
  to { opacity: 1; transform: translateY(0); }
}
@media (prefers-reduced-motion: reduce) {
  .mm-cmd { animation: none; }
  .mm-cmd-field, #go, .mm-ex button { transition: none; }
}
@media (max-width: 680px) {
  .mm-cmd-inner { padding: 14px; }
  .mm-cmd-head { align-items: flex-start; flex-direction: column; }
  .mm-cmd-field { align-items: stretch; flex-wrap: wrap; padding: 8px; }
  .mm-cmd-ico { display: none; }
  #q { flex-basis: 100%; min-height: 34px; }
  #go { width: 100%; }
}
"""

_COMMAND_JS = """
export default function (component) {
  const { data, parentElement, setStateValue, setTriggerValue } = component
  const root = parentElement.querySelector("#root")
  if (!root) return
  if (root.dataset.bound === "1") {
    hydrate(root, data, setStateValue)
    return
  }
  root.dataset.bound = "1"
  const q = root.querySelector("#q")
  const go = root.querySelector("#go")
  const modes = root.querySelector("#modes")
  const show = root.querySelector("#show")
  const examples = root.querySelector("#examples")
  const meta = root.querySelector("#meta")

  const modeList = (data && data.modes) || []
  modes.innerHTML = ""
  modeList.forEach((m) => {
    const b = document.createElement("button")
    b.type = "button"
    b.textContent = m
    b.dataset.mode = m
    b.setAttribute("role", "tab")
    b.onclick = () => {
      setStateValue("mode", m)
      ;[...modes.querySelectorAll("button")].forEach((el) => {
        el.setAttribute("aria-selected", el.dataset.mode === m ? "true" : "false")
      })
    }
    modes.appendChild(b)
  })

  const showOpts = (data && data.showOptions) || [25, 50, 100, 250, "All"]
  show.innerHTML = ""
  showOpts.forEach((opt) => {
    const o = document.createElement("option")
    o.value = String(opt)
    o.textContent = "Show " + String(opt)
    show.appendChild(o)
  })

  examples.innerHTML = ""
  ;((data && data.examples) || []).forEach((ex) => {
    const b = document.createElement("button")
    b.type = "button"
    const label = (typeof ex === "string") ? ex : (ex.label || "")
    const full = (typeof ex === "string") ? ex : (ex.query || label)
    b.textContent = label
    b.onclick = () => {
      q.value = full
      setStateValue("query", full)
      setTriggerValue("submitted", { query: full, mode: currentMode(), show: show.value })
    }
    examples.appendChild(b)
  })

  // Deliberately NOT wired to oninput: pushing state (and triggering a Streamlit
  // rerun) on every keystroke means a fast typist outruns the round-trip and
  // characters arrive out of order or get dropped when the rerun re-hydrates this
  // field mid-word. The query only needs to reach the server on submit (Enter /
  // Run / an example pill) -- all three already read q.value directly -- so typing
  // stays entirely local until then.
  q.onkeydown = (e) => {
    if (e.key === "Enter") {
      e.preventDefault()
      setStateValue("query", q.value)
      setTriggerValue("submitted", { query: q.value, mode: currentMode(), show: show.value })
    }
  }
  go.onclick = () => {
    setStateValue("query", q.value)
    setTriggerValue("submitted", { query: q.value, mode: currentMode(), show: show.value })
  }
  show.onchange = () => setStateValue("show", show.value)

  function currentMode() {
    const sel = modes.querySelector('button[aria-selected="true"]')
    return (sel && sel.dataset.mode) || "hybrid"
  }

  hydrate(root, data, setStateValue)
}

function hydrate(root, data, setStateValue) {
  const q = root.querySelector("#q")
  const modes = root.querySelector("#modes")
  const show = root.querySelector("#show")
  const meta = root.querySelector("#meta")
  if (!q || !modes || !show) return
  const next = (data && data.query) != null ? String(data.query) : ""
  if (q.value !== next && document.activeElement !== q) q.value = next
  const mode = (data && data.mode) || "hybrid"
  ;[...modes.querySelectorAll("button")].forEach((el) => {
    el.setAttribute("aria-selected", el.dataset.mode === mode ? "true" : "false")
  })
  const showVal = (data && data.show) != null ? String(data.show) : "50"
  if (show.value !== showVal) show.value = showVal
  if (meta) meta.textContent = (data && data.meta) || ""
}
"""

_LIST_HTML = """
<div class="mm-list" id="root">
  <div class="mm-list-head">
    <div>
      <div class="mm-list-kicker">Results</div>
      <div class="mm-list-title" id="title">Candidates</div>
    </div>
    <div class="mm-list-meta" id="meta"></div>
  </div>
  <div class="mm-list-body" id="body"></div>
  <div class="mm-list-empty" id="empty" hidden>No candidates match these filters.</div>
</div>
"""

_LIST_CSS = """
:host, .mm-list {
  --ink: #09090B; --muted: #71717A; --faint: #A1A1AA; --line: #E4E4E7;
  --surface: #FFFFFF; --elev: #FAFAFA; --accent: #4F46E5; --accent-soft: #EEF2FF;
  --accent-text: #3730A3; --warn: #D97706; --ok: #16A34A;
  font-family: "Plus Jakarta Sans", -apple-system, "Segoe UI", system-ui, sans-serif;
  color: var(--ink);
}
*, *::before, *::after { box-sizing: border-box; }
.mm-list {
  background: var(--surface); border: 1px solid var(--line); border-radius: 8px;
  box-shadow: 0 12px 36px rgba(9,9,11,.06); overflow: hidden;
  animation: rise .4s cubic-bezier(.22,1,.36,1) .05s both;
}
.mm-list-head {
  display: flex; align-items: flex-end; justify-content: space-between;
  gap: 12px; padding: 16px 18px 12px; border-bottom: 1px solid var(--line);
  background: linear-gradient(180deg, #FAFAFA 0%, #fff 100%);
}
.mm-list-kicker {
  font-size: 0.66rem; font-weight: 750; letter-spacing: .12em;
  text-transform: uppercase; color: var(--faint); margin-bottom: 3px;
}
.mm-list-title { font-size: 1.05rem; font-weight: 700; letter-spacing: 0; }
.mm-list-meta {
  font-family: "JetBrains Mono", ui-monospace, Menlo, monospace;
  font-size: 0.68rem; color: var(--faint);
}
.mm-list-body {
  display: flex; flex-direction: column;
  max-height: 612px; overflow: auto;
}
.mm-row {
  display: grid; grid-template-columns: 42px minmax(0,1.5fr) minmax(180px,.9fr) auto auto;
  gap: 12px; align-items: center; padding: 12px 16px;
  border-bottom: 1px solid #F4F4F5; cursor: pointer;
  transition: background .15s ease, transform .12s ease;
}
.mm-row:last-child { border-bottom: 0; }
.mm-row:hover {
  background: linear-gradient(90deg, var(--accent-soft), transparent 70%);
}
.mm-row:hover .mm-open {
  opacity: 1; transform: translateX(0);
}
.mm-av {
  width: 38px; height: 38px; border-radius: 12px;
  display: flex; align-items: center; justify-content: center;
  font-size: 0.72rem; font-weight: 750; color: #fff; letter-spacing: 0;
  box-shadow: 0 4px 10px rgba(9,9,11,.12);
}
.mm-name { font-weight: 650; font-size: 0.9rem; letter-spacing: 0; }
.mm-sub {
  font-size: 0.74rem; color: var(--muted); margin-top: 2px;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.mm-tags { display: flex; flex-wrap: wrap; gap: 4px; }
.mm-tag {
  font-size: 0.66rem; font-weight: 650; color: var(--accent-text);
  background: var(--accent-soft); border: 1px solid #C7D2FE;
  border-radius: 999px; padding: 2px 8px;
}
.mm-tag.muted { color: var(--muted); background: var(--elev); border-color: var(--line); }
.mm-tag.warn { color: #92400E; background: #FFFBEB; border-color: #FDE68A; }
.mm-score {
  font-family: "JetBrains Mono", ui-monospace, Menlo, monospace;
  font-size: 0.78rem; font-weight: 650; color: var(--ink); text-align: right;
  min-width: 52px;
}
.mm-score span { display: block; font-size: 0.62rem; color: var(--faint); font-weight: 500; }
.mm-actions { display: flex; gap: 6px; }
.mm-actions button, .mm-open {
  border: 1px solid var(--line); background: #fff; border-radius: 9px;
  font: inherit; font-size: 0.72rem; font-weight: 650; padding: 6px 10px;
  cursor: pointer; color: var(--ink);
  transition: border-color .15s ease, background .15s ease, transform .12s ease;
}
.mm-actions button:hover { border-color: var(--accent); background: var(--accent-soft); }
.mm-actions button.on {
  background: #FFFBEB; border-color: #FDE68A; color: #92400E;
}
.mm-open {
  opacity: 0; transform: translateX(4px); color: var(--accent-text);
  background: var(--accent-soft); border-color: #C7D2FE;
}
.mm-list-empty {
  padding: 36px 18px; text-align: center; color: var(--muted); font-size: 0.88rem;
}
@keyframes rise {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}
@media (max-width: 820px) {
  .mm-row { grid-template-columns: 42px minmax(0,1fr) auto; }
  .mm-tags, .mm-score { display: none; }
}
@media (prefers-reduced-motion: reduce) {
  .mm-list { animation: none; }
  .mm-row, .mm-open { transition: none; }
}
"""

_LIST_JS = """
export default function (component) {
  const { data, parentElement, setTriggerValue } = component
  const root = parentElement.querySelector("#root")
  if (!root) return
  const title = root.querySelector("#title")
  const meta = root.querySelector("#meta")
  const body = root.querySelector("#body")
  const empty = root.querySelector("#empty")
  const rows = (data && data.rows) || []
  title.textContent = (data && data.title) || "Candidates"
  meta.textContent = (data && data.meta) || ""
  body.innerHTML = ""
  empty.hidden = rows.length > 0
  const palette = ["#4F46E5","#0EA5E9","#16A34A","#D97706","#7C3AED","#DB2777","#0891B2"]
  rows.forEach((r, i) => {
    const row = document.createElement("div")
    row.className = "mm-row"
    row.tabIndex = 0
    const av = document.createElement("div")
    av.className = "mm-av"
    av.style.background = palette[i % palette.length]
    av.textContent = (r.initials || "?").slice(0, 2)
    const main = document.createElement("div")
    main.innerHTML = '<div class="mm-name"></div><div class="mm-sub"></div>'
    main.querySelector(".mm-name").textContent = r.name || r.id
    main.querySelector(".mm-sub").textContent = r.subtitle || ""
    const tags = document.createElement("div")
    tags.className = "mm-tags"
    ;(r.tags || []).forEach((t) => {
      const s = document.createElement("span")
      s.className = "mm-tag" + (t.tone === "warn" ? " warn" : t.tone === "muted" ? " muted" : "")
      s.textContent = t.label
      tags.appendChild(s)
    })
    const score = document.createElement("div")
    score.className = "mm-score"
    if (r.score != null && r.score !== "") {
      score.innerHTML = String(r.score) + "<span>score</span>"
    } else {
      score.innerHTML = "—<span>rank</span>"
    }
    const actions = document.createElement("div")
    actions.className = "mm-actions"
    const sl = document.createElement("button")
    sl.type = "button"
    sl.textContent = r.listed ? "Listed" : "Shortlist"
    if (r.listed) sl.classList.add("on")
    sl.onclick = (e) => {
      e.stopPropagation()
      setTriggerValue("action", { type: "shortlist", id: r.id })
    }
    const open = document.createElement("button")
    open.type = "button"
    open.className = "mm-open"
    open.textContent = "Open"
    open.onclick = (e) => {
      e.stopPropagation()
      setTriggerValue("action", { type: "open", id: r.id })
    }
    actions.appendChild(sl)
    actions.appendChild(open)
    row.appendChild(av)
    row.appendChild(main)
    row.appendChild(tags)
    row.appendChild(score)
    row.appendChild(actions)
    row.onclick = () => setTriggerValue("action", { type: "open", id: r.id })
    row.onkeydown = (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault()
        setTriggerValue("action", { type: "open", id: r.id })
      }
    }
    body.appendChild(row)
  })
}
"""

_command = None
_list = None
_bound_manager = None


def _active_manager_id() -> int:
    try:
        from streamlit.components.v2.get_bidi_component_manager import (
            get_bidi_component_manager,
        )
        return id(get_bidi_component_manager())
    except Exception:
        return 0


def _renderers():
    """Register against the *current* BidiComponentManager (AppTest has its own)."""
    global _command, _list, _bound_manager
    mid = _active_manager_id()
    if _command is not None and _list is not None and _bound_manager == mid:
        return _command, _list
    _command = st.components.v2.component(
        "mm_command_bar",
        html=_COMMAND_HTML,
        css=_COMMAND_CSS,
        js=_COMMAND_JS,
    )
    _list = st.components.v2.component(
        "mm_candidate_list",
        html=_LIST_HTML,
        css=_LIST_CSS,
        js=_LIST_JS,
    )
    _bound_manager = mid
    return _command, _list


def _ss(key: str) -> dict:
    return st.session_state[key] if key in st.session_state else {}


def command_bar(
    *,
    query: str,
    mode: str,
    show: str | int,
    examples: list,
    meta: str = "",
    key: str = "cc_command",
) -> Any:
    """Linear-style command surface. Syncs query/mode/show into session_state."""

    def _noop() -> None:
        return None

    mount, _ = _renderers()
    result = mount(
        key=key,
        data={
            "query": query,
            "mode": mode,
            "show": str(show),
            "modes": [],
            "showOptions": [25, 50, 100, 250, "All"],
            "examples": examples,
            "meta": meta,
        },
        default={"query": query, "mode": mode, "show": str(show)},
        on_query_change=_noop,
        on_mode_change=_noop,
        on_show_change=_noop,
        on_submitted_change=_noop,
        width="stretch",
    )

    if getattr(result, "query", None) is not None:
        st.session_state.query = result.query
    st.session_state.retrieval_mode = "hybrid"
    if getattr(result, "show", None) is not None:
        raw = result.show
        st.session_state.f_show_n = "All" if str(raw) == "All" else int(raw)
    submitted = getattr(result, "submitted", None)
    if isinstance(submitted, dict):
        if "query" in submitted:
            st.session_state.query = submitted["query"]
        st.session_state.retrieval_mode = "hybrid"
        if submitted.get("show") is not None:
            raw = submitted["show"]
            st.session_state.f_show_n = "All" if str(raw) == "All" else int(raw)
    return result


def candidate_list(
    rows: list[dict],
    *,
    title: str,
    meta: str = "",
    key: str = "cc_candidate_list",
    on_action: Callable[[dict], None] | None = None,
) -> Any:
    """Dense interactive result rows (avatar, tags, score, open/shortlist)."""

    def _on_action() -> None:
        payload = None
        if key in st.session_state:
            blk = st.session_state[key]
            if isinstance(blk, dict):
                payload = blk["action"] if "action" in blk else None
            else:
                payload = getattr(blk, "action", None)
        if on_action and isinstance(payload, dict):
            on_action(payload)

    _, mount = _renderers()
    return mount(
        key=key,
        data={"rows": rows, "title": title, "meta": meta},
        on_action_change=_on_action if on_action else (lambda: None),
        width="stretch",
        height=700,
    )
