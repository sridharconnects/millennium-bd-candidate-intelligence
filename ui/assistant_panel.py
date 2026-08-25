"""The chat assistant's UI: a full-height dock on the right of the page, toggled from
a chat icon in the header -- the same shape as VS Code's Copilot panel or this very
tool's own chat pane, not a cramped sidebar afterthought.

Message state is kept as ONE list, in the exact Anthropic API format
(`st.session_state.chat_api_messages`) -- not duplicated into a separate "display"
list. Chat bubbles are derived from it at render time by filtering out tool_use /
tool_result blocks, so there is a single source of truth and no way for the two to
drift out of sync.
"""
from __future__ import annotations

import html

import streamlit as st

from millennium.assistant import AssistantUnavailable, run_turn

SUGGESTIONS = {
    "Most senior in APAC?": "Who is the most senior candidate based in APAC?",
    "Shortlist & open a profile": "Shortlist Ryan Patel and open his profile",
    "Summarize the pool": "Give me a quick summary of the candidate pool",
    "Filter by strategy": "Filter to long/short equity candidates",
}


def _block_text(block) -> str | None:
    """A block is either an Anthropic SDK object (assistant turns) or a plain dict
    (the tool_result turns this module appends itself) -- handle both shapes."""
    btype = getattr(block, "type", None) or (block.get("type") if isinstance(block, dict) else None)
    if btype == "text":
        return getattr(block, "text", None) or (block.get("text") if isinstance(block, dict) else None)
    return None


def _display_turns(messages: list[dict]) -> list[tuple[str, str]]:
    """-> [(role, text)] for whichever turns actually carry visible text -- a
    pure-tool-call assistant turn (no text block) or a tool_result turn is skipped
    here, not shown as an empty bubble."""
    out = []
    for m in messages:
        texts = [t for c in m["content"] for t in [_block_text(c)] if t]
        if texts:
            out.append((m["role"], " ".join(texts)))
    return out


def init_state() -> None:
    st.session_state.setdefault("chat_api_messages", [])
    st.session_state.setdefault("chat_open", False)
    st.session_state.setdefault("chat_last_actions", [])


def render_toggle_button() -> None:
    """The header icon that opens/closes the dock -- 'press chat in top' the way VS
    Code's own chat toggle works, rather than a switch buried in the sidebar."""
    is_open = st.session_state.chat_open
    label = "✕  Chat" if is_open else "💬  Chat"
    if st.button(label, key="chat_toggle_btn", type="primary" if is_open else "secondary",
                width="stretch",
                help="A real Claude tool-calling loop over this exact workspace -- it "
                     "can search, filter, open a candidate, manage the shortlist, and "
                     "answer questions using live pool data. It cannot delete a record "
                     "or approve a review; those stay behind your own click."):
        st.session_state.chat_open = not is_open
        st.rerun()


def _send(user_text: str, pool: list, store) -> None:
    st.session_state.chat_api_messages.append(
        {"role": "user", "content": [{"type": "text", "text": user_text}]})
    try:
        with st.spinner("Thinking…"):
            result = run_turn(st.session_state.chat_api_messages, pool, store)
    except AssistantUnavailable as e:
        st.session_state.chat_api_messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": str(e)}]})
        st.session_state.chat_last_actions = []
    else:
        st.session_state.chat_last_actions = result.actions
    # Always rerun: this is what makes any state the tool loop changed (page, query,
    # shortlist) actually show up -- both in this panel's own freshly-rendered history
    # and on whatever page the assistant switched to.
    st.rerun()


def render_chat_dock(pool: list, store) -> None:
    """The full right-hand panel. Call only when `st.session_state.chat_open` is True,
    inside the column reserved for it -- see app.py's routing section."""
    with st.container(key="chat_dock"):
        head, clear, close = st.columns([0.72, 0.14, 0.14])
        with head:
            st.markdown(
                '<div class="mm-chat-title">💬 Assistant'
                '<span class="mm-chat-live"><span class="dot"></span>live</span></div>',
                unsafe_allow_html=True)
        with clear:
            if st.button("↺", key="chat_clear_btn", help="Start a new conversation",
                        width="stretch"):
                st.session_state.chat_api_messages = []
                st.session_state.chat_last_actions = []
                st.rerun()
        with close:
            if st.button("✕", key="chat_close_btn", help="Close the assistant",
                        width="stretch"):
                st.session_state.chat_open = False
                st.rerun()
        st.markdown(
            '<div class="mm-sub" style="margin:-4px 0 10px 0">Real tool-calling over '
            'this workspace — search, filter, open a candidate, manage the shortlist. '
            'It never deletes a record or approves a review.</div>',
            unsafe_allow_html=True)

        turns = _display_turns(st.session_state.chat_api_messages)
        if not turns:
            st.markdown(
                '<div style="text-align:center;padding:28px 10px 18px 10px">'
                '<div style="font-size:2rem;line-height:1">💬</div>'
                '<div style="font-weight:700;font-size:1.0rem;margin-top:8px">'
                'How can I help?</div>'
                '<div class="mm-sub" style="margin-top:2px">Ask about candidates, run a '
                'search, or manage the shortlist.</div></div>', unsafe_allow_html=True)
            picked = st.pills("Suggestions", list(SUGGESTIONS.keys()),
                              label_visibility="collapsed", key="chat_suggestion_pill")
            if picked:
                _send(SUGGESTIONS[picked], pool, store)
        else:
            with st.container(height=560):
                for role, text in turns:
                    with st.chat_message("user" if role == "user" else "assistant"):
                        st.markdown(text)

        if st.session_state.chat_last_actions:
            chips = "".join(
                f'<span class="mm-chip mm-chip-plain">🛠 {html.escape(a)}</span>'
                for a in st.session_state.chat_last_actions)
            st.markdown(f'<div style="margin-top:4px">{chips}</div>',
                       unsafe_allow_html=True)

        user_text = st.chat_input("Ask or tell it what to do…", key="chat_dock_input")
    if user_text:
        _send(user_text, pool, store)
