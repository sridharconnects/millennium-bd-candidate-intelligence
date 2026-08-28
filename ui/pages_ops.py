"""Review / Analytics / System — the pages that make the tool auditable."""
from __future__ import annotations

import html
import json
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from millennium import taxonomy as tx
from millennium.agents.base import registry_table
from millennium.agents.insight import (coverage_gaps, data_quality, distributions,
                                       skill_cooccurrence)
from millennium.config import SETTINGS
from . import components as C
from . import llm_panels
from . import theme


def _byid(pool):
    return {p.candidate_id: p for p in pool}


def _review_fields(p):
    fields = {"Name": ("sensitive.full_name", p.sensitive.full_name),
              "Email": ("sensitive.email", p.sensitive.email),
              "Phone": ("sensitive.phone", p.sensitive.phone),
              "Headline": ("headline", p.headline),
              "Location": ("location_current", p.location_current)}
    for i, e in enumerate(p.employment[:5]):
        fields[f"Employer #{i+1}"] = (f"employment[{i}].employer_raw", e.employer_raw)
        fields[f"Title #{i+1}"] = (f"employment[{i}].title_raw", e.title_raw)
    return fields


def _review_initials(name: str) -> str:
    return "".join(part[0] for part in name.replace("(", " ").split()
                   if part[:1].isalpha())[:2].upper() or "?"


def _review_severity(p):
    score = (len(p.quality.validation_flags) * 2 +
             p.quality.abstention_count +
             len(p.quality.review_reasons))
    if p.quality.completeness < 0.6 or score >= 6:
        return "High", "danger"
    if p.quality.evidence_coverage < 0.85 or score >= 3:
        return "Medium", "warn"
    return "Low", "ok"


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value or 0.0)))


def _review_lane(p) -> str:
    if p.quality.validation_flags:
        return "Correct contradiction"
    if p.quality.abstention_count:
        return "Fill missing proof"
    if p.quality.evidence_coverage < 0.85:
        return "Check source spans"
    return "Approve quality"


def _review_action_from_text(text: str) -> tuple[str, str]:
    low = text.lower()
    if "timeline" in low or "date" in low or "degree" in low:
        return (
            "Verify chronology",
            "Compare education dates and role dates against the source document, then correct the wrong date or approve the exception.",
        )
    if "contact" in low or "email" in low or "phone" in low or "name" in low:
        return (
            "Confirm identity fields",
            "Check the highlighted source span and correct name, email, phone, or location if extraction merged the wrong value.",
        )
    if "span" in low or "evidence" in low or "unverified" in low:
        return (
            "Check source evidence",
            "Open the evidence beside the field and only approve values that are directly supported by the uploaded resume.",
        )
    if "abstain" in low or "missing" in low:
        return (
            "Fill missing proof",
            "Use the field selector to find abstained values; add the source-backed value or leave it blank if the resume never states it.",
        )
    return (
        "Resolve validation flag",
        "Read the routing note, inspect the source evidence, then correct the extracted value or approve it as a known exception.",
    )


def _review_action_items(p, fields=None) -> list[tuple[str, str, str]]:
    items: list[tuple[str, str, str]] = []
    for flag in p.quality.validation_flags:
        title, action = _review_action_from_text(flag)
        items.append((title, flag, action))
    for reason in p.quality.review_reasons:
        if reason not in {item[1] for item in items}:
            title, action = _review_action_from_text(reason)
            items.append((title, reason, action))
    if fields:
        for label, (_path, tracked) in fields.items():
            if tracked.validation_status in {"abstained", "conflicted", "unverified"}:
                title, action = _review_action_from_text(tracked.validation_status)
                note = tracked.notes[0] if tracked.notes else tracked.validation_status
                items.append((f"Review {label}", note, action))
    if not items and p.quality.evidence_coverage < 0.85:
        items.append((
            "Check source evidence",
            f"Evidence coverage is {p.quality.evidence_coverage:.0%}.",
            "Inspect the field evidence before approving the profile.",
        ))
    if not items:
        items.append((
            "Approve profile",
            "No unresolved extracted field is flagged.",
            "Approve the profile or add a correction if you spot an issue in the source document.",
        ))
    return items[:5]


def _render_ai_json_panel(data: dict, title: str) -> None:
    if data.get("_notice"):
        st.info(data["_notice"])
    source = "Uses LLM: generated" if data.get("_llm_generated") else "LLM feature: fallback"
    st.markdown(
        f'<div class="mm-ai-panel-head"><span>{html.escape(title)}</span>'
        f'<b>{html.escape(source)}</b></div>',
        unsafe_allow_html=True)
    for key, value in data.items():
        if key.startswith("_"):
            continue
        label = key.replace("_", " ").title()
        if isinstance(value, list):
            st.markdown(f"**{label}**")
            if value and isinstance(value[0], dict):
                for item in value[:8]:
                    bits = [f"**{html.escape(str(k).replace('_', ' ').title())}:** "
                            f"{html.escape(str(v))}" for k, v in item.items() if v not in ("", [], None)]
                    st.markdown("- " + " · ".join(bits) if bits else "-")
            else:
                for item in value[:8]:
                    st.markdown(f"- {html.escape(str(item))}")
        elif value not in ("", None, []):
            st.markdown(f"**{label}**  \n{html.escape(str(value))}")


def _render_review_summary(queue, profiles, store):
    total_abstained = sum(p.quality.abstention_count for p in profiles)
    total_flags = sum(len(p.quality.validation_flags) for p in profiles)
    mean_complete = (sum(p.quality.completeness for p in profiles) /
                     max(1, len(profiles)))
    mean_evidence = (sum(p.quality.evidence_coverage for p in profiles) /
                     max(1, len(profiles)))
    st.markdown(
        '<div class="mm-review-hero">'
        '<div><div class="mm-page-title">Review workbench</div>'
        '<div class="mm-page-sub">A focused triage desk for deciding what is wrong, '
        'what needs correction, and what can safely move forward.</div></div>'
        '<div class="mm-review-flow">'
        '<span>Route</span><b></b><span>Correct</span><b></b><span>Approve</span>'
        '</div></div>',
        unsafe_allow_html=True)
    st.markdown(
        '<div class="mm-review-metrics">'
        f'<div><b>{len(queue)}</b><span>in queue</span><small>of {len(profiles)} records</small></div>'
        f'<div><b>{total_flags}</b><span>validation flags</span><small>possible contradictions</small></div>'
        f'<div><b>{total_abstained}</b><span>abstained fields</span><small>not guessed by the agent</small></div>'
        f'<div><b>{len(store.audit_trail())}</b><span>audit entries</span><small>review history</small></div>'
        f'<div><b>{mean_complete:.0%}</b><span>mean complete</span><small>whole pool</small></div>'
        f'<div><b>{mean_evidence:.0%}</b><span>mean evidenced</span><small>source-backed fields</small></div>'
        '</div>',
        unsafe_allow_html=True)


def _render_review_queue(queue):
    ids = [p.candidate_id for p in queue]
    st.session_state.setdefault("review_selected", ids[0])
    if st.session_state["review_selected"] not in ids:
        st.session_state["review_selected"] = ids[0]

    st.markdown(
        f'<div class="mm-panel-heading"><span>Review queue</span>'
        f'<b>{len(queue)} waiting</b></div>',
        unsafe_allow_html=True)
    q = st.text_input("Find in queue", "", placeholder="Search name, role or reason",
                      key="review_queue_search", label_visibility="collapsed")
    needle = q.strip().lower()
    visible = []
    for rec in queue:
        haystack = " ".join([rec.display_name(st.session_state.blind),
                             str(rec.headline.display() or ""),
                             " ".join(rec.quality.review_reasons),
                             " ".join(rec.quality.validation_flags)]).lower()
        if not needle or needle in haystack:
            visible.append(rec)
    if not visible:
        st.markdown('<div class="mm-review-empty">No queued record matches that search.</div>',
                    unsafe_allow_html=True)
        return
    if st.session_state["review_selected"] not in {rec.candidate_id for rec in visible}:
        st.session_state["review_selected"] = visible[0].candidate_id
        st.rerun()

    for rec in visible:
        is_sel = rec.candidate_id == st.session_state["review_selected"]
        sev, tone = _review_severity(rec)
        name = rec.display_name(st.session_state.blind)
        initials = _review_initials(name)
        reason = rec.quality.review_reasons[0] if rec.quality.review_reasons else _review_lane(rec)
        todo = _review_action_items(rec)[0][0]
        st.markdown(
            f'<div class="mm-review-pick {tone}{" is-selected" if is_sel else ""}">'
            f'<div class="mm-review-avatar">{html.escape(initials)}</div>'
            f'<div class="mm-review-pick-body">'
            f'<div class="mm-review-pick-top"><span>{html.escape(name)}</span>'
            f'<b>{html.escape(sev)}</b></div>'
            f'<div class="mm-review-pick-reason">{html.escape(reason)}</div>'
            f'<div class="mm-review-pick-action"><b>Next</b>{html.escape(todo)}</div>'
            f'<div class="mm-review-pick-meta">'
            f'{len(rec.quality.review_reasons)} reasons · '
            f'{len(rec.quality.validation_flags)} flags · '
            f'{rec.quality.abstention_count} abstained</div></div></div>',
            unsafe_allow_html=True)
        if st.button("Selected" if is_sel else "Review", key=f"rev_pick_{rec.candidate_id}",
                     type="primary" if is_sel else "secondary", disabled=is_sel,
                     width="stretch"):
            st.session_state["review_selected"] = rec.candidate_id
            st.rerun()


def _render_review_case(p, fields, store, client):
    name = p.display_name(st.session_state.blind)
    initials = _review_initials(name)
    sev, tone = _review_severity(p)
    headline = p.headline.display() or "No headline extracted"
    source = p.provenance.source_file or p.doc_id
    problem = (p.quality.review_reasons[0] if p.quality.review_reasons
               else "Record is waiting for a reviewer decision.")
    action_items = _review_action_items(p, fields)
    action_html = "".join(
        f'<div><b>{html.escape(title)}</b><span>{html.escape(detail)}</span>'
        f'<small>{html.escape(action)}</small></div>'
        for title, detail, action in action_items[:3])
    st.markdown(
        f'<div class="mm-review-case-head {tone}">'
        f'<div class="mm-review-avatar xl">{html.escape(initials)}</div>'
        f'<div class="mm-review-case-title"><div class="mm-review-case-kicker">'
        f'{html.escape(_review_lane(p))}</div>'
        f'<h2>{html.escape(name)}</h2>'
        f'<p>{html.escape(str(headline))}</p>'
        f'<div class="mm-review-source">{html.escape(source)}</div></div>'
        f'<div class="mm-review-severity"><span>{html.escape(sev)}</span>'
        f'<small>severity</small></div></div>'
        f'<div class="mm-review-problem"><b>What needs attention</b>'
        f'<span>{html.escape(problem)}</span></div>'
        f'<div class="mm-review-action-grid">{action_html}</div>',
        unsafe_allow_html=True)

    C.provenance_banner(p)

    st.markdown('<div class="mm-panel-heading"><span>Decision checklist</span>'
                '<b>colour = severity</b></div>', unsafe_allow_html=True)
    c1, c2 = st.columns([0.54, 0.46])
    with c1:
        st.markdown('<div class="mm-review-subtitle">Why this was routed here</div>',
                    unsafe_allow_html=True)
        if p.quality.review_reasons:
            st.markdown(theme.flag_list(p.quality.review_reasons), unsafe_allow_html=True)
        else:
            st.caption("No specific routing reasons recorded.")
    with c2:
        st.markdown('<div class="mm-review-subtitle">Immediate correction targets</div>',
                    unsafe_allow_html=True)
        flagged_fields = [(lbl, t) for lbl, (_path, t) in fields.items()
                          if t.validation_status in {"abstained", "conflicted", "unverified"}]
        if flagged_fields:
            for lbl, t in flagged_fields[:4]:
                note = t.notes[0] if t.notes else t.validation_status
                st.markdown(theme.flag_card(f"{lbl}: {note}"), unsafe_allow_html=True)
        elif p.quality.validation_flags:
            for flag in p.quality.validation_flags[:4]:
                title, action = _review_action_from_text(flag)
                st.markdown(theme.flag_card(f"{title}: {action}"), unsafe_allow_html=True)
        else:
            st.caption("No correctable field here is abstained, conflicted or unverified.")

    with st.container(border=True, key="review_editor_card"):
        st.markdown('<div class="mm-panel-heading"><span>Correct or approve a field</span>'
                    '<b>source evidence stays visible</b></div>', unsafe_allow_html=True)
        plan_key = f"review_ai_plan_{p.candidate_id}"
        C.llm_callout(
            "Review plan",
            "Uses the LLM to suggest what is wrong, what evidence to inspect, and "
            "which fields need correction. Reviewers still make the decision.",
            stage="review_plan")
        if st.button("Generate LLM review plan", icon=":material/auto_awesome:",
                     key=f"review_ai_plan_btn_{p.candidate_id}", width="stretch"):
            st.session_state[plan_key] = llm_panels.review_plan(
                client, p, fields, blind=st.session_state.blind)
        if st.session_state.get(plan_key):
            _render_ai_json_panel(st.session_state[plan_key], "LLM review plan")
        pick = st.selectbox("Field to review", list(fields),
                            key=f"review_field_{p.candidate_id}")
        path, t = fields[pick]
        safe = (path.replace("[", "_").replace("]", "").replace(".", "_"))
        left, right = st.columns([0.42, 0.58])
        with left:
            st.markdown(C.tracked_value(t, pick), unsafe_allow_html=True)
            for note in t.notes:
                st.caption(f"· {note}")
            new = st.text_input("Corrected value", str(t.value or ""), key=f"corr_{path}")
            reviewer = st.text_input("Reviewer", "bd.analyst", key="reviewer")
            a, b = st.columns(2)
            if a.button("Save correction", type="primary", width="stretch",
                        key=f"save_corr_{p.candidate_id}_{safe}"):
                old = t.value
                t.value, t.normalized_value = new, new
                t.confidence = 1.0
                t.extraction_method = "human"
                t.validation_status = "human_corrected"
                t.notes.append(f"corrected by {reviewer}")
                store.log_review(p.candidate_id, path, old, new, reviewer, "correct")
                st.session_state.corrections[f"{p.candidate_id}:{path}"] = new
                st.success("Correction saved to the audit log. This is also a training signal.")
            if b.button("Approve as-is", width="stretch",
                        key=f"approve_corr_{p.candidate_id}_{safe}"):
                store.log_review(p.candidate_id, path, t.value, t.value, reviewer, "approve")
                st.success("Approved.")
        with right:
            st.markdown('<div class="mm-review-subtitle">Source evidence</div>',
                        unsafe_allow_html=True)
            C.evidence_for(t, p)


def _render_review_context(p, fields, store, index):
    st.markdown('<div class="mm-panel-heading"><span>Record context</span>'
                '<b>trust & history</b></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="mm-review-trust-grid">'
        f'<div><b>{p.quality.completeness:.0%}</b><span>complete</span></div>'
        f'<div><b>{p.quality.evidence_coverage:.0%}</b><span>evidenced</span></div>'
        f'<div><b>{p.quality.abstention_count}</b><span>abstained</span></div>'
        f'<div><b>{len(p.quality.validation_flags)}</b><span>flags</span></div>'
        '</div>',
        unsafe_allow_html=True)
    st.progress(_clamp01(p.quality.completeness), text="Profile completeness")
    st.progress(_clamp01(p.quality.evidence_coverage), text="Evidence coverage")

    abstained_here = [(lbl, t) for lbl, (_path, t) in fields.items()
                      if t.validation_status == "abstained"]
    st.markdown(
        f'<div class="mm-review-subtitle">Abstained fields — {len(abstained_here)} of '
        f'{len(fields)} checked here ({p.quality.abstention_count} total on this record)</div>',
        unsafe_allow_html=True)
    if abstained_here:
        for lbl, t in abstained_here:
            reason = t.notes[0] if t.notes else "no reason recorded"
            st.markdown(theme.flag_card(f"{lbl}: {reason}"), unsafe_allow_html=True)
    else:
        st.caption("No profile/contact/title field abstained in this correction set.")

    if p.quality.validation_flags:
        with st.expander(f"{len(p.quality.validation_flags)} validation flag(s)", expanded=True):
            st.markdown(theme.flag_list(p.quality.validation_flags), unsafe_allow_html=True)

    trail = store.audit_trail(p.candidate_id)
    st.markdown(
        f'<div class="mm-panel-heading compact"><span>Audit trail for this record</span>'
        f'<b>{len(trail)} entr{"y" if len(trail) == 1 else "ies"}</b></div>',
        unsafe_allow_html=True)
    action_style = {
        "correct": ("#6D28D9", "#F5F3FF", "corrected"),
        "approve": ("#15803D", "#F0FDF4", "approved as-is"),
        "gdpr_delete": ("#B91C1C", "#FEF2F2", "erased"),
    }
    if trail:
        for row in trail[:8]:
            fg, bg, word = action_style.get(row["action"],
                                            ("#71717A", "#F4F4F5", str(row["action"])))
            new_val = "" if row["action"] == "gdpr_delete" else \
                html.escape(str(row["new_value"] or "")[:58])
            st.markdown(
                f'<div class="mm-review-audit-row" style="--fg:{fg};--bg:{bg};--bd:{fg}33">'
                f'<div><b>{html.escape(row["field"])}</b> {html.escape(word)}'
                + (f' <span>{new_val}</span>' if new_val else "")
                + f'</div><small>{html.escape(str(row["reviewer"]))} · '
                f'{html.escape(str(row["created_at"]))}</small></div>',
                unsafe_allow_html=True)
        with st.expander("Full audit table"):
            st.dataframe(pd.DataFrame(store.audit_trail())[
                ["created_at", "candidate_id", "field", "action", "reviewer", "new_value"]],
                width="stretch", hide_index=True)
    else:
        st.caption("No corrections recorded for this candidate yet.")

    with st.expander("Right to erasure (GDPR Art. 17 / CCPA)"):
        st.caption("Deletion removes database rows, the search index entry, vector index "
                   "state, and the on-disk profile.")
        if st.button("Delete this candidate permanently", type="secondary",
                     key=f"delete_review_{p.candidate_id}"):
            res = store.delete_candidate(p.candidate_id, index)
            st.json(res)
            st.cache_resource.clear()
            st.warning("Erased. Reload to refresh the pool.")


# ============================================================================ REVIEW
def render_review(profiles, synth, pool, index, index_manifest, manifest, store,
                  client, bench, evals):
    queue = [p for p in profiles if p.quality.needs_human_review]
    st.markdown(
        '<div class="mm-review-hero compact">'
        '<div><div class="mm-page-title">Review workbench</div>'
        '<div class="mm-page-sub">Queue, source evidence, correction, approve. '
        'Only the fields that need a human eye stay in the main workspace.</div></div>'
        f'<div class="mm-review-flow"><span>{len(queue)} waiting</span></div></div>',
        unsafe_allow_html=True)

    if not queue:
        C.empty_state("Nothing needs review",
                      "Every profile in the current pool is complete enough to publish.")
        return

    ids = [p.candidate_id for p in queue]
    byid = _byid(pool)
    st.session_state.setdefault("review_selected", ids[0])
    if st.session_state["review_selected"] not in ids:
        st.session_state["review_selected"] = ids[0]

    p = byid[st.session_state["review_selected"]]
    fields = _review_fields(p)

    left, center = st.columns([0.24, 0.76], gap="medium")
    with left:
        with st.container(border=True, key="review_queue_panel"):
            _render_review_queue(queue)
    with center:
        with st.container(border=True, key="review_case_panel"):
            _render_review_case(p, fields, store, client)
        with st.expander("Audit, quality, and erasure controls", expanded=False):
            _render_review_context(p, fields, store, index)


# ========================================================================= ANALYTICS
def render_analytics(profiles, synth, pool, index, index_manifest, manifest, store,
                     client, bench, evals):
    C.synthetic_banner(len(synth) if st.session_state.include_synthetic else 0)
    dist = distributions(pool).output or {}
    dq = data_quality(pool).output or {}
    gaps = coverage_gaps(pool).output or {}

    C.section_header("Pool intelligence",
                     subtitle="Coverage, gaps, and extraction quality across the working set")
    k = st.columns(5)
    C.kpi(k[0], dq.get("candidates", 0), "Candidates", "in working pool")
    C.kpi(k[1], f"{dq.get('mean_completeness', 0):.0%}", "Mean complete",
          delta=None)
    C.kpi(k[2], f"{dq.get('mean_evidence_coverage', 0):.0%}", "Evidenced")
    C.kpi(k[3], dq.get("total_abstentions", 0), "Abstained", "refused, not guessed",
          colour=theme.WARNING if dq.get("total_abstentions", 0) else theme.SUCCESS)
    C.kpi(k[4], dq.get("needs_review", 0), "Need review",
          colour=theme.WARNING if dq.get("needs_review", 0) else theme.INK)

    tabs = st.tabs(["LLM memo", "Distributions", "Coverage gaps", "Skills",
                    "Data quality", "Export"])

    with tabs[0]:
        st.markdown("##### LLM pool memo")
        C.llm_callout(
            "Pool memo",
            "Uses the LLM to write an advisory narrative from aggregate pool data. "
            "It does not see candidate contact details or make hiring decisions.",
            stage="analytics_memo")
        memo_key = "analytics_ai_memo"
        if st.button("Generate LLM pool memo", icon=":material/auto_awesome:",
                     key="analytics_ai_memo_btn"):
            st.session_state[memo_key] = llm_panels.analytics_memo(
                client, pool, dq, dist, gaps)
        if st.session_state.get(memo_key):
            _render_ai_json_panel(st.session_state[memo_key], "LLM pool memo")
        else:
            st.markdown(
                '<div class="mm-ai-empty">Generate a concise readout of the current '
                'pool, coverage risks, recommended searches, and data quality actions.'
                '</div>',
                unsafe_allow_html=True)

    with tabs[1]:
        pairs = [("region", "Geographic market"), ("strategy", "Investment strategy"),
                 ("sector", "Sector coverage"), ("seniority", "Seniority level"),
                 ("experience_band", "Experience"), ("approach", "Investment approach"),
                 ("feeder", "Feeder path"), ("employer_tier", "Employer tier")]
        for i in range(0, len(pairs), 2):
            cols = st.columns(2)
            for j, (key, title) in enumerate(pairs[i:i + 2]):
                data = dist.get(key, {})
                if not data:
                    continue
                df = pd.DataFrame({"label": list(data), "n": list(data.values())})
                fig = px.bar(df, x="n", y="label", orientation="h", text="n",
                             color_discrete_sequence=[theme.SERIES[(i + j) % len(theme.SERIES)]],
                             height=max(200, 32 * len(df) + 90))
                fig.update_layout(title=title, title_font_size=13, showlegend=False,
                                  margin=dict(l=0, r=10, t=34, b=0),
                                  yaxis_title=None, xaxis_title=None, xaxis_visible=False,
                                  yaxis=dict(autorange="reversed"),
                                  plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                  font=dict(size=11))
                fig.update_traces(textposition="outside", cliponaxis=False)
                cols[j].plotly_chart(theme.polish_fig(fig), width="stretch",
                                     config={"displayModeBar": False})

        st.markdown("**Strategy × sector coverage**")
        cells = []
        for p in pool:
            for s in p.strategies:
                for sec in p.sectors:
                    cells.append({"strategy": tx.display("strategy", s.label),
                                  "sector": tx.display("sector", sec.label)})
        if cells:
            piv = (pd.DataFrame(cells).value_counts().reset_index(name="n")
                   .pivot(index="strategy", columns="sector", values="n").fillna(0))
            fig = px.imshow(piv, text_auto=True, aspect="auto",
                            color_continuous_scale=["#EEF2FF", theme.ACCENT],
                            height=90 + 34 * len(piv))
            fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), coloraxis_showscale=False,
                              font=dict(size=11), xaxis_title=None, yaxis_title=None)
            st.plotly_chart(theme.polish_fig(fig), width="stretch", config={"displayModeBar": False})

    with tabs[2]:
        st.markdown("##### Where you cannot currently hire")
        st.caption("The inverse of a distribution chart. A recruiter already knows most "
                   "of the pool is equity research; what they need is the list of "
                   "requisitions this pool cannot fill.")
        g = gaps.get("gaps", [])
        empty = [x for x in g if x["count"] == 0]
        thin = [x for x in g if x["count"] > 0]
        a, b = st.columns(2)
        with a:
            st.markdown(f"**No coverage at all — {len(empty)} dimension(s)**")
            for x in empty[:22]:
                st.markdown(f'<div class="mm-danger" style="padding:5px 10px;margin:3px 0">'
                            f'{html.escape(x["label"])} <span class="mm-sub">'
                            f'({x["dimension"]})</span></div>', unsafe_allow_html=True)
        with b:
            st.markdown(f"**Thin coverage — {len(thin)} dimension(s)**")
            for x in thin[:22]:
                st.markdown(f'<div class="mm-warn" style="padding:5px 10px;margin:3px 0">'
                            f'{html.escape(x["label"])} — {x["count"]} candidate(s) '
                            f'<span class="mm-sub">({x["dimension"]})</span></div>',
                            unsafe_allow_html=True)
        st.markdown("**Strongest cells**")
        strong = gaps.get("strongest_cells", [])
        if strong:
            st.dataframe(pd.DataFrame(strong), width="stretch", hide_index=True)

    with tabs[3]:
        a, b = st.columns([0.45, 0.55])
        with a:
            data = dist.get("skill", {})
            deep = dist.get("skill_deep", {})
            df = pd.DataFrame([{"skill": k, "mentions": v, "applied or core": deep.get(k, 0)}
                               for k, v in data.items()])
            if not df.empty:
                fig = px.bar(df.head(20).melt(id_vars="skill"), x="value", y="skill",
                             color="variable", orientation="h", barmode="overlay",
                             color_discrete_sequence=["#CBD5E1", theme.ACCENT],
                             height=34 * min(20, len(df)) + 110)
                fig.update_layout(title="Skill depth — listed vs actually used",
                                  title_font_size=13, margin=dict(l=0, r=0, t=34, b=0),
                                  yaxis=dict(autorange="reversed"), yaxis_title=None,
                                  xaxis_title=None, legend_title=None,
                                  legend=dict(orientation="h", y=1.06),
                                  plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                  font=dict(size=11))
                st.plotly_chart(theme.polish_fig(fig), width="stretch", config={"displayModeBar": False})
        with b:
            co = skill_cooccurrence(pool).output or []
            if co:
                st.markdown("**Capabilities that travel together**")
                st.caption("Useful when writing a requisition: asking for a combination "
                           "nobody in the pool has is how a search stalls.")
                st.dataframe(pd.DataFrame(co).head(20), width="stretch",
                             hide_index=True)

    with tabs[4]:
        a, b = st.columns(2)
        with a:
            rows = [{"candidate": p.display_name(st.session_state.blind),
                     "completeness": p.quality.completeness,
                     "evidence coverage": p.quality.evidence_coverage,
                     "extraction quality": p.quality.extraction_quality,
                     "abstentions": p.quality.abstention_count}
                    for p in pool]
            df = pd.DataFrame(rows).sort_values("completeness")
            fig = px.bar(df.melt(id_vars="candidate",
                                 value_vars=["completeness", "evidence coverage",
                                             "extraction quality"]),
                         x="value", y="candidate", color="variable", barmode="group",
                         orientation="h", color_discrete_sequence=theme.SERIES,
                         height=42 * len(df) + 120)
            fig.update_layout(title="Per-record data quality", title_font_size=13,
                              margin=dict(l=0, r=0, t=34, b=0), yaxis_title=None,
                              xaxis_title=None, legend_title=None,
                              legend=dict(orientation="h", y=1.05),
                              plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                              font=dict(size=11))
            st.plotly_chart(theme.polish_fig(fig), width="stretch", config={"displayModeBar": False})
        with b:
            st.markdown("**Pool-level honesty metrics**")
            st.json(dq)
            st.caption("`total_abstentions` is reported as prominently as any accuracy "
                       "number. In a hiring product a fabricated employer is worse than "
                       "a blank, so a refusal is a success state, not a failure.")

    with tabs[5]:
        _render_export_tab()


def _render_export_tab() -> None:
    """The case study's own deliverable #2, made reachable from inside the product.

    The files existed on disk from the first pipeline run, but nothing in the app
    ever pointed at them -- a BD user with no terminal access had no way to get this
    data out. Every file here is read straight from `data/exports/`, the exact
    artefact `scripts/run_pipeline.py` wrote; nothing is regenerated on the fly, so
    what you preview is byte-identical to what you download.
    """
    exp = SETTINGS.paths.exports
    files = [
        ("candidates.json", "Full-fidelity export — every field with its evidence, "
                            "confidence, and validation status.", "json"),
        ("candidates.csv", "One row per candidate, flattened for Excel. A blank cell "
                           "plus its own `*_status` column distinguishes 'abstained' "
                           "from 'missing' — never collapsed into one meaning.", "csv"),
        ("employment.csv", "One row per role across every candidate.", "csv"),
        ("education.csv", "One row per degree across every candidate.", "csv"),
        ("skills.csv", "One row per skill, with depth (core/applied/mentioned) and "
                       "evidence count.", "csv"),
        ("evidence.csv", "Every source span behind every claim in the export above — "
                         "candidate, document, character offsets, match quality.", "csv"),
    ]
    missing = [name for name, _d, _k in files if not (exp / name).exists()]
    if missing:
        st.markdown(
            f'<div class="mm-warn">Missing from <code>data/exports/</code>: '
            f'{", ".join(missing)}. Run <code>python scripts/run_pipeline.py</code> to '
            f'generate them.</div>', unsafe_allow_html=True)

    present = [(n, d, k) for n, d, k in files if (exp / n).exists()]
    if not present:
        return

    total_bytes = sum((exp / n).stat().st_size for n, _d, _k in present)
    k = st.columns(3)
    C.kpi(k[0], len(present), "export files ready")
    C.kpi(k[1], f"{total_bytes/1024:.0f} KB", "total size")
    C.kpi(k[2], SETTINGS.schema_version, "schema version")
    generated = datetime.fromtimestamp((exp / present[0][0]).stat().st_mtime)
    st.caption(f"data/exports/ · generated {generated.strftime('%Y-%m-%d %H:%M')}")

    for name, desc, kind in present:
        path = exp / name
        raw = path.read_bytes()
        st.markdown(f'<div class="mm-card"><div class="mm-row" '
                   f'style="justify-content:space-between">'
                   f'<span class="mm-name mm-mono">{name}</span>'
                   f'<span class="mm-sub mm-mono">{len(raw)/1024:.1f} KB</span></div>'
                   f'<div class="mm-sub">{html.escape(desc)}</div></div>',
                   unsafe_allow_html=True)
        c1, c2 = st.columns([0.82, 0.18])
        with c1:
            with st.expander(f"Preview {name}"):
                if kind == "csv":
                    st.dataframe(pd.read_csv(path), width="stretch", height=260)
                else:
                    payload = json.loads(raw)
                    st.caption(f"{payload.get('count', '?')} candidates · schema "
                              f"{payload.get('schema_version', '?')} · showing the "
                              f"first record in full, the rest have the identical shape")
                    cands = payload.get("candidates", [])
                    st.json(cands[0] if cands else payload, expanded=False)
        with c2:
            mime = "application/json" if kind == "json" else "text/csv"
            st.download_button("Download", raw, file_name=name, mime=mime,
                              key=f"dl_export_{name}", width="stretch")


# ============================================================================ SYSTEM
def render_system(profiles, synth, pool, index, index_manifest, manifest, store,
                  client, bench, evals):
    tabs = st.tabs(["Pipeline", "Retrieval ablation", "Evaluation", "Calibration",
                    "Scalability", "Fairness", "Cost & index", "Agents"])

    with tabs[0]:
        st.markdown(
            '<div class="mm-banner">The full agent-by-agent pipeline diagrams — '
            'parsing, search, requisition matching, analytics, and the chat '
            'assistant loop, with every node labelled by exactly what produces it '
            '(LLM API call / local model / rule-based) — now live on their own '
            '<b>Workflow</b> page in the sidebar, not nested in a System tab. This '
            'tab keeps the raw run manifest and per-document outcome table.</div>',
            unsafe_allow_html=True)
        st.markdown("##### Pipeline run manifest")
        st.caption("Every artefact in this repo traces back to one of these runs.")
        st.json(manifest or {"note": "no manifest found"})
        st.markdown("##### Per-document outcome")
        rows = [{"candidate": p.display_name(st.session_state.blind),
                 "source": p.provenance.source_file if p.provenance else "",
                 "type": p.provenance.file_type if p.provenance else "",
                 "roles": len(p.employment), "education": len(p.education),
                 "skills": len(p.skills), "completeness": p.quality.completeness,
                 "evidence": p.quality.evidence_coverage,
                 "abstained": p.quality.abstention_count,
                 "flags": len(p.quality.validation_flags),
                 "review": p.quality.needs_human_review,
                 "cost_usd": p.provenance.cost_usd if p.provenance else 0}
                for p in profiles]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    with tabs[1]:
        st.markdown("##### Retrieval ablation")
        st.caption("Measured on the labelled query set with graded relevance (0–3). "
                   "The hybrid claim is tested here rather than asserted.")
        ab = (evals or {}).get("ablation")
        if not ab:
            st.info("Run `python scripts/run_eval.py` to populate this table.")
        else:
            df = pd.DataFrame(ab)
            st.dataframe(df, width="stretch", hide_index=True)
            metric = "ndcg@10" if "ndcg@10" in df.columns else df.columns[1]
            fig = px.bar(df, x="mode", y=metric, text=metric,
                         color_discrete_sequence=[theme.ACCENT], height=300)
            fig.update_traces(texttemplate="%{text:.3f}", textposition="outside",
                              cliponaxis=False)
            fig.update_layout(margin=dict(l=0, r=0, t=20, b=0), yaxis_title=metric,
                              xaxis_title=None, plot_bgcolor="rgba(0,0,0,0)",
                              paper_bgcolor="rgba(0,0,0,0)", font=dict(size=11))
            st.plotly_chart(theme.polish_fig(fig), width="stretch", config={"displayModeBar": False})

    with tabs[2]:
        st.markdown("##### Extraction accuracy vs hand-labelled gold")
        ex = (evals or {}).get("extraction")
        if not ex:
            st.info("Run `python scripts/run_eval.py` to populate this.")
        else:
            k = st.columns(5)
            C.kpi(k[0], f"{ex.get('macro_f1', 0):.3f}", "macro F1", "across fields")
            C.kpi(k[1], f"{ex.get('hallucination_rate', 0):.1%}", "hallucination rate",
                  "target 0", theme.ACCENT if not ex.get("hallucination_rate") else "#B91C1C")
            C.kpi(k[2], f"{ex.get('abstention_rate', 0):.1%}", "abstention rate",
                  "refused, not guessed")
            C.kpi(k[3], f"{ex.get('schema_validity', 0):.0%}", "schema valid")
            C.kpi(k[4], f"{ex.get('evidence_coverage', 0):.0%}", "evidence coverage")
            if ex.get("per_field"):
                st.dataframe(pd.DataFrame(ex["per_field"]), width="stretch",
                             hide_index=True)
            if ex.get("rule_vs_llm"):
                st.markdown("**Rule layer vs LLM, per field**")
                st.caption("Reported honestly. Where regex beats the model — dates, "
                           "emails — that is the finding, not something to hide.")
                st.dataframe(pd.DataFrame(ex["rule_vs_llm"]), width="stretch",
                             hide_index=True)

    with tabs[3]:
        st.markdown("##### Is the confidence number honest?")
        st.caption("A confidence score that does not track observed accuracy is worse "
                   "than no score: it invites trust precisely where trust is "
                   "unwarranted. Every grounded field prediction is bucketed by its "
                   "predicted confidence and compared against the hand-labelled gold set.")
        cal = (evals or {}).get("calibration")
        if not cal or not cal.get("reliability_curve"):
            st.info("Run `python scripts/run_eval.py` to populate this.")
        else:
            k = st.columns(4)
            ece = cal["ece"]
            C.kpi(k[0], f"{ece:.3f}", "ECE", "expected calibration error",
                  theme.ACCENT if ece < 0.10 else "#B45309")
            C.kpi(k[1], f"{cal['brier']:.3f}", "Brier score", "lower is better")
            C.kpi(k[2], cal["n_samples"], "predictions scored")
            C.kpi(k[3], cal["verdict"], "verdict",
                  colour=theme.ACCENT if ece < 0.10 else "#B45309")

            curve = pd.DataFrame(cal["reliability_curve"])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                     name="perfect calibration",
                                     line=dict(dash="dot", color="#94A3B8")))
            fig.add_trace(go.Scatter(x=curve["mean_confidence"],
                                     y=curve["observed_accuracy"],
                                     mode="lines+markers+text",
                                     text=[f"n={n}" for n in curve["n"]],
                                     textposition="top center", name="observed",
                                     line=dict(color=theme.ACCENT, width=2),
                                     marker=dict(size=10)))
            fig.update_layout(title="Reliability diagram", title_font_size=13,
                              xaxis_title="predicted confidence",
                              yaxis_title="observed accuracy",
                              xaxis=dict(range=[0, 1.02]), yaxis=dict(range=[0, 1.02]),
                              height=380, margin=dict(l=0, r=0, t=36, b=0),
                              plot_bgcolor="rgba(0,0,0,0)",
                              paper_bgcolor="rgba(0,0,0,0)", font=dict(size=11),
                              legend=dict(orientation="h", y=1.12))
            a, b = st.columns([0.55, 0.45])
            a.plotly_chart(theme.polish_fig(fig), width="stretch", config={"displayModeBar": False})
            with b:
                st.markdown("**Per bucket**")
                st.dataframe(curve, width="stretch", hide_index=True)
                st.markdown("**Per field**")
                st.dataframe(pd.DataFrame(cal["per_field"]), width="stretch",
                             hide_index=True)
            st.markdown(f'<div class="mm-banner">{html.escape(cal["interpretation"])}'
                        f'</div>', unsafe_allow_html=True)
            st.caption("Points below the diagonal mean over-confidence — the system "
                       "claims more certainty than it earns. Points above mean it is "
                       "under-selling itself, which costs recall.")

    with tabs[4]:
        st.markdown("##### Scalability")
        st.caption("Deliverable #5 answered with measurements rather than prose.")
        if not bench:
            st.info("Run `python scripts/run_benchmark.py` to populate this.")
        else:
            df = pd.DataFrame(bench.get("points", []))
            if not df.empty:
                a, b = st.columns(2)
                fig = go.Figure()
                for col, nm in (("p50_ms", "p50"), ("p95_ms", "p95")):
                    if col in df:
                        fig.add_trace(go.Scatter(x=df["n_candidates"], y=df[col],
                                                 mode="lines+markers", name=nm))
                fig.update_layout(title="Search latency vs corpus size", title_font_size=13,
                                  xaxis_title="candidates indexed", yaxis_title="ms",
                                  height=320, margin=dict(l=0, r=0, t=34, b=0),
                                  plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                  font=dict(size=11),
                                  colorway=[theme.ACCENT, "#B45309"])
                a.plotly_chart(theme.polish_fig(fig), width="stretch", config={"displayModeBar": False})
                fig2 = px.line(df, x="n_candidates", y="index_build_ms", markers=True,
                               height=320, color_discrete_sequence=[theme.SERIES[1]])
                fig2.update_layout(title="Index build time", title_font_size=13,
                                   xaxis_title="candidates indexed", yaxis_title="ms",
                                   margin=dict(l=0, r=0, t=34, b=0),
                                   plot_bgcolor="rgba(0,0,0,0)",
                                   paper_bgcolor="rgba(0,0,0,0)", font=dict(size=11))
                b.plotly_chart(theme.polish_fig(fig2), width="stretch", config={"displayModeBar": False})
                st.dataframe(df, width="stretch", hide_index=True)
            st.markdown("**Migration triggers** — thresholds, not adjectives")
            st.dataframe(pd.DataFrame(bench.get("migration_triggers", [])),
                         width="stretch", hide_index=True)

    with tabs[5]:
        st.markdown("##### Fairness by construction")
        st.markdown(
            "Protected attributes live in a separate `SensitiveAttributes` model. The "
            "scoring function's signature accepts only `ScorableProfile`, which "
            "**structurally has no field** that could carry one — no name, no contact "
            "details, no address, no marital status, no nationality, no hobbies. This "
            "is a property of the type system, checked by a test, not a promise.")
        from millennium.schema import ScorableProfile, SensitiveAttributes
        a, b = st.columns(2)
        a.markdown("**Fields the scorer CAN see**")
        a.code("\n".join(sorted(ScorableProfile.model_fields)), language="text")
        b.markdown("**Fields quarantined from it**")
        b.code("\n".join(sorted(SensitiveAttributes.model_fields)), language="text")
        fa = (evals or {}).get("fairness")
        if fa:
            st.markdown("**Counterfactual name-swap audit**")
            st.caption("Every candidate's name is replaced with names of different "
                       "apparent origin and the ranking is re-run. Mean rank change is "
                       "zero by construction — the scorer never received the name.")
            st.json(fa)
        st.markdown(
            '<div class="mm-banner">Regulatory note: employment-screening tools are '
            'classified high-risk under the EU AI Act (Annex III), and NYC Local Law 144 '
            'requires an annual independent bias audit for automated employment decision '
            'tools. This product is positioned as decision support with a human approving '
            'every shortlist, which is the posture those regimes expect — but a '
            'production deployment would still need the formal audit, a model card, and '
            'candidate-facing notice.</div>', unsafe_allow_html=True)

    with tabs[6]:
        a, b = st.columns(2)
        with a:
            st.markdown("**Index manifest**")
            st.caption("Validated on load. A mismatch between the model that built the "
                       "index and the model querying it is refused outright — silent "
                       "embedding drift degrades results in a way nobody notices.")
            st.json(index_manifest)
        with b:
            st.markdown("**LLM cache**")
            st.json(client.cache_stats())
            st.markdown("**Cost**")
            st.json({"parse_cost_usd": manifest.get("cost_usd", 0),
                     "cost_per_resume_usd": manifest.get("cost_per_doc_usd", 0),
                     "llm_calls": manifest.get("llm_calls", 0),
                     "cache_hits": manifest.get("llm_cache_hits", 0),
                     "retrieval_cost_usd": 0.0,
                     "note": "retrieval, ranking and the entire UI run locally at zero "
                             "marginal cost; the LLM is used once per document at ingest"})

    with tabs[7]:
        st.markdown("##### Agent registry")
        st.caption("Every subagent here does real work and is independently testable. "
                   "Consolidation was deliberate — the cut log is in DECISIONS.md.")
        df = pd.DataFrame(registry_table())
        st.dataframe(df, width="stretch", hide_index=True,
                     column_config={"description": st.column_config.TextColumn(width="large")})
        st.caption(f"{len(df)} subagents across {df['agent'].nunique()} agents.")
