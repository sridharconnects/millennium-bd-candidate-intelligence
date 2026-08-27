"""Overview — the platform's own hero/ontology diagram, embedded live in the app.

This was originally shipped only as a standalone Artifact link, which meant nobody
looking at the running app itself ever saw it — a real gap, not a stylistic choice.
It now lives in the repo (`assets/ontology.html`) and renders here via an iframe, so
it is part of the actual product, not a separate thing you have to be told to go find.

Deliberately its own page rather than a header banner on Search: it is a considered,
self-contained dark "ops-deck" design (see the file's own comments for the palette and
typography rationale), and dropping it inline above the dense light-themed workspace
would fight that design rather than let it read as intentional.

The hero needs an exit, the same way Palantir's own reference has a "Get Started"
button riding top-right of theirs: a page that only shows how the mechanism works and
never invites you into it is a poster, not a product. That's the CTA button below --
the one element in the whole app that's allowed to visibly draw the eye, because its
entire job is to move someone from watching the diagram to using the real workspace.
"""
from __future__ import annotations

import streamlit as st

from millennium.config import SETTINGS

ONTOLOGY_PATH = SETTINGS.paths.root / "assets" / "ontology.html"


def _open_app_button(key: str, label: str = "Open the App →") -> bool:
    """A real Streamlit button that navigates into Search -- not a link inside the
    static iframe, which is sandboxed and cannot reach back into session_state.

    The pulsing animation is scoped in `theme.py` to `.st-key-cta_top` specifically
    (Streamlit derives that class from this widget's own `key`), so only the ONE
    top-of-page button we want drawing the eye actually pulses -- the calmer
    bottom-of-page copy uses a different key and stays still by construction, not by
    a second flag threaded through here.
    """
    return st.button(label, key=key, type="primary", width="stretch")


def render_overview(profiles, synth, pool, index, index_manifest, manifest, store,
                    client, bench, evals) -> None:
    head, cta = st.columns([0.72, 0.28], vertical_alignment="center")
    with head:
        st.caption("How ten resumes become a queryable, evidence-grounded candidate "
                  "graph — every claim traceable to its source span.")
    with cta:
        st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)
        if _open_app_button("cta_top"):
            st.session_state.page = "Search"
            st.rerun()
        st.markdown('<div class="mm-sub" style="text-align:right;font-size:0.72rem">'
                   'jump straight to the working search screen</div>',
                   unsafe_allow_html=True)

    if not ONTOLOGY_PATH.exists():
        st.markdown(
            '<div class="mm-warn">assets/ontology.html is missing from this checkout '
            '— the diagram cannot render. It should be committed alongside the rest '
            'of the repo.</div>', unsafe_allow_html=True)
        return

    html = ONTOLOGY_PATH.read_text(encoding="utf8")
    st.components.v1.html(html, height=2050, scrolling=True)

    st.divider()
    st.caption("The pipeline stages, evidence threads, and relationship edges above "
              "are not illustrative — they are the real agent sequence, the real "
              "data contracts passed between stages, and the real taxonomy this "
              "repository ships. See the System page for the live registry behind it.")

    end_l, end_r = st.columns([0.72, 0.28])
    with end_r:
        if _open_app_button("cta_bottom"):
            st.session_state.page = "Search"
            st.rerun()
