from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from app_functions import (
    prepare_data,
    protocol_name,
    plot_protocol_strip,
    plot_bout_count_rewards,
    plot_stacked_lick_counts,
    plot_histogram_kde_failures,
    plot_kde_failures_by_session,
    plot_regression_rewards_failures_and_slope,
    build_session_plot_rewards_vs_failures,
    build_session_plot_failure_distribution,
    get_month_options,
    month_to_label,
    build_month_day_list,
    get_session_date_to_protocol,
)

# =============================================================================
# CONFIG
# =============================================================================
DB_PATH = Path(r"\\SynoINVIBE_Caze\INVIBE_team_Cazettes\data\database\full_db_all_rigs.feather")
DEFAULT_VERSION = "1"
DEFAULT_REWARD_CUT = 7
DEFAULT_FAILURE_MAX = 25

st.set_page_config(
    page_title="Mouse Behavior Dashboard",
    page_icon="🐭",
    layout="wide"
)

st.markdown("""
<style>
.block-container {
    padding-top: 0.85rem;
    padding-bottom: 1.8rem;
    max-width: 1540px;
}
h1, h2, h3 {
    letter-spacing: -0.02em;
}
div[data-testid="stMetric"] {
    background-color: #FAFBFC;
    border-radius: 14px;
}

/* calendar visual */
.calendar-wrap {
    display: flex;
    flex-direction: column;
    gap: 6px;
}
.calendar-row {
    display: flex;
    align-items: center;
    gap: 6px;
}
.calendar-month-label {
    width: 82px;
    min-width: 82px;
    color: #667085;
    font-weight: 600;
    font-size: 0.92rem;
}
.calendar-square {
    width: 24px;
    height: 24px;
    border-radius: 6px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 11px;
    font-weight: 700;
    border: 1px solid rgba(0,0,0,0.05);
    box-sizing: border-box;
}
.calendar-empty {
    background: #F1F4F8;
    color: #A3ACB9;
}
.calendar-t1 {
    background: #F6E06E;
    color: #4E420A;
}
.calendar-t2 {
    background: #90C5FF;
    color: #143A66;
}
.calendar-t3 {
    background: #F2A093;
    color: #51211A;
}
.calendar-selected {
    outline: 2px solid #223248;
    outline-offset: 1px;
}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# UI HELPERS
# =============================================================================
def render_plot_card(title, fig, help_text=None, use_container_width=True):
    with st.container(border=True):
        st.subheader(title)
        if help_text:
            st.caption(help_text)
        st.pyplot(fig, use_container_width=use_container_width, clear_figure=True)
    plt.close(fig)


def render_metric_row(items):
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        col.metric(label, value, border=True)


def render_visual_calendar(valid_sessions, selected_date):
    session_date_to_protocol = get_session_date_to_protocol(valid_sessions)
    valid_dates = sorted(session_date_to_protocol.keys())
    months = get_month_options(valid_dates)

    html_parts = ["<div class='calendar-wrap'>"]

    for year, month in months:
        month_days = build_month_day_list(year, month)
        row = [
            f"<div class='calendar-row'><div class='calendar-month-label'>{month_to_label(year, month)}</div>"
        ]

        for d in month_days:
            proto = session_date_to_protocol.get(d, None)
            cls = "calendar-empty"
            if proto == 1:
                cls = "calendar-t1"
            elif proto == 2:
                cls = "calendar-t2"
            elif proto == 3:
                cls = "calendar-t3"

            selected_cls = " calendar-selected" if d == selected_date else ""
            row.append(f"<span class='calendar-square {cls}{selected_cls}'>{d.day}</span>")

        row.append("</div>")
        html_parts.append("".join(row))

    html_parts.append("</div>")
    st.markdown("".join(html_parts), unsafe_allow_html=True)


# =============================================================================
# LOAD DATA
# =============================================================================
if not DB_PATH.exists():
    st.error(f"Base introuvable : {DB_PATH}")
    st.stop()


@st.cache_data(show_spinner=False)
def load_all(path):
    return prepare_data(path)


with st.spinner("Loading data..."):
    df, session_cmap = load_all(DB_PATH)

if df.empty:
    st.error("La base est vide.")
    st.stop()


# =============================================================================
# DEFAULT MOUSE = MOST RECENT SESSION
# =============================================================================
latest_by_mouse = (
    df.groupby("Mouse_ID", dropna=True)["Date"]
    .max()
    .sort_values()
)
default_mouse = latest_by_mouse.index[-1] if len(latest_by_mouse) > 0 else None

mouse_options = sorted(df["Mouse_ID"].dropna().unique().tolist())
if not mouse_options:
    st.error("Aucune souris trouvée dans la base.")
    st.stop()

if default_mouse not in mouse_options:
    default_mouse = mouse_options[0]


# =============================================================================
# SESSION STATE
# =============================================================================
if "mouse_id" not in st.session_state:
    st.session_state.mouse_id = default_mouse

if "view_mode" not in st.session_state:
    st.session_state.view_mode = "Overview"

if st.session_state.mouse_id not in mouse_options:
    st.session_state.mouse_id = default_mouse


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.header("Navigation")

    selected_mouse = st.selectbox(
        "Mouse",
        mouse_options,
        index=mouse_options.index(st.session_state.mouse_id),
        key="sidebar_mouse"
    )

    selected_view = st.radio(
        "View",
        ["Overview", "Session focus"],
        index=0 if st.session_state.view_mode == "Overview" else 1,
        key="sidebar_view"
    )

mouse_changed = selected_mouse != st.session_state.mouse_id
view_changed = selected_view != st.session_state.view_mode

st.session_state.mouse_id = selected_mouse
st.session_state.view_mode = selected_view

mouse_id = st.session_state.mouse_id
view_mode = st.session_state.view_mode


# =============================================================================
# FILTERED DATA
# =============================================================================
df_mouse_all = df[df["Mouse_ID"] == mouse_id].sort_values("Date").copy()
df_mouse_v1 = df_mouse_all[df_mouse_all["Version"] == DEFAULT_VERSION].copy()

valid_sessions_for_state = (
    df_mouse_v1
    .sort_values("Date")
    .drop_duplicates(subset=["Date_norm"], keep="last")
    .copy()
)
valid_dates_for_state = valid_sessions_for_state["Date_norm"].dt.date.tolist()

if "selected_date" not in st.session_state:
    st.session_state.selected_date = valid_dates_for_state[-1] if valid_dates_for_state else None

if mouse_changed:
    st.session_state.selected_date = valid_dates_for_state[-1] if valid_dates_for_state else None

if st.session_state.selected_date not in valid_dates_for_state and valid_dates_for_state:
    st.session_state.selected_date = valid_dates_for_state[-1]

st.title(f"🐭 {mouse_id}")

# page placeholder pour éviter les résidus visuels d'une vue à l'autre
page = st.empty()

with page.container():
    # =============================================================================
    # OVERVIEW
    # =============================================================================
    if view_mode == "Overview":
        with st.spinner("Loading overview..."):
            render_metric_row([
                ("Sessions", len(df_mouse_all)),
                ("First date", df_mouse_all["Date"].min().strftime("%Y-%m-%d") if not df_mouse_all.empty else "-"),
                ("Last date", df_mouse_all["Date"].max().strftime("%Y-%m-%d") if not df_mouse_all.empty else "-"),
            ])

            fig_strip = plot_protocol_strip(df, mouse_id)
            fig_bouts = plot_bout_count_rewards(df, mouse_id)
            fig_licks = plot_stacked_lick_counts(df, mouse_id)
            fig_hist = plot_histogram_kde_failures(df, mouse_id)
            fig_kde = plot_kde_failures_by_session(df, mouse_id, session_cmap)
            fig_reg = plot_regression_rewards_failures_and_slope(
                df=df,
                mouse=mouse_id,
                session_cmap=session_cmap,
                max_reward=7,
                max_failure=30,
                min_valid_bouts=100
            )

            if fig_strip is not None:
                render_plot_card("Session type distribution", fig_strip, use_container_width=True)

            if fig_bouts is not None:
                render_plot_card("Bout count + rewarded licks", fig_bouts)

            if fig_licks is not None:
                render_plot_card("Rewarded / non-rewarded / invalid licks", fig_licks)

            col1, col2 = st.columns(2)

            with col1:
                if fig_hist is not None:
                    render_plot_card("Global distribution of consecutive failures", fig_hist)

            with col2:
                if fig_kde is not None:
                    render_plot_card("KDE of consecutive failures by task session", fig_kde)

            if fig_reg is not None:
                render_plot_card("Reward / failure regression + slope over time", fig_reg)

    # =============================================================================
    # SESSION FOCUS
    # =============================================================================
    else:
        valid_sessions = (
            df_mouse_v1
            .sort_values("Date")
            .drop_duplicates(subset=["Date_norm"], keep="last")
            .copy()
        )

        valid_dates = valid_sessions["Date_norm"].dt.date.tolist()

        if not valid_dates:
            st.warning("Aucune session valide trouvée pour la version 1.")
            st.stop()

        if st.session_state.selected_date not in valid_dates:
            st.session_state.selected_date = valid_dates[-1]

        current_selected_date = st.session_state.selected_date

        with st.container(border=True):
            st.subheader("Session calendar")
            st.caption("Calendrier visuel + sélection simple par date.")

            render_visual_calendar(
                valid_sessions=valid_sessions,
                selected_date=current_selected_date
            )

            st.markdown("")

            date_options = [d.isoformat() for d in valid_dates]
            current_idx = date_options.index(current_selected_date.isoformat())

            chosen = st.selectbox(
                "Session date",
                options=date_options,
                index=current_idx,
                key="session_date_select"
            )

            chosen_date = pd.to_datetime(chosen).date()
            if chosen_date != st.session_state.selected_date:
                st.session_state.selected_date = chosen_date
                st.rerun()

        selected_date = st.session_state.selected_date

        subset = df[
            (df["Mouse_ID"] == mouse_id) &
            (df["Date_norm"].dt.date == selected_date) &
            (df["Version"] == DEFAULT_VERSION)
        ].copy()

        if subset.empty:
            st.warning("Session non trouvée.")
            st.stop()

        session = subset.iloc[0]
        date_str = pd.Timestamp(selected_date).strftime("%Y-%m-%d")

        valid_bouts = np.asarray(session["Correct Bouts"], dtype=bool)
        n_valid_bouts = int(np.sum(valid_bouts)) if len(valid_bouts) > 0 else 0

        with st.spinner("Loading session..."):
            render_metric_row([
                ("Date", date_str),
                ("Probas", str(session["Probas"]) if "Probas" in session.index else "-"),
                ("Protocol", protocol_name(session["Protocol"]) if "Protocol" in session.index else "-"),
                ("Valid bouts", n_valid_bouts),
            ])

            fig1 = build_session_plot_rewards_vs_failures(
                session=session,
                mouse_id=mouse_id,
                date_str=date_str,
                reward_cut=DEFAULT_REWARD_CUT
            )
            fig2 = build_session_plot_failure_distribution(
                session=session,
                failure_xlim=(0, DEFAULT_FAILURE_MAX),
                reward_cut=DEFAULT_REWARD_CUT
            )

            colA, colB = st.columns(2)

            with colA:
                render_plot_card("Rewards vs failures", fig1)

            with colB:
                render_plot_card("Failure distribution", fig2)