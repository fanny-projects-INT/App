import ast
import colorsys
import calendar

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from scipy.stats import gaussian_kde
from sklearn.linear_model import LinearRegression
from matplotlib.colors import LinearSegmentedColormap


# =============================================================================
# STYLE / COLORS
# =============================================================================
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "medium",
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "grid.alpha": 0.16,
})

# Palette unique, claire et cohérente partout
YELLOW = "#F6E06E"
BLUE = "#90C5FF"
RED = "#F2A093"

COLORS = {
    "navy": "#223248",
    "blue": BLUE,
    "blue_light": "#F2F8FF",
    "purple": "#8574C8",
    "purple_light": "#EEE9FA",
    "green": "#5DA884",
    "green_light": "#E5F2EB",
    "red": RED,
    "red_light": "#FFF0EC",
    "yellow": YELLOW,
    "yellow_light": "#FFF9D9",
    "gray": "#748091",
    "gray_light": "#F1F4F8",
    "grid": "#D9E1EA",
}

PROTOCOL_COLORS = {
    1: (0.96, 0.88, 0.42),   # jaune
    2: (0.56, 0.77, 1.00),   # bleu
    3: (0.95, 0.63, 0.58),   # rouge
}

PROTOCOL_HEX = {
    1: YELLOW,   # Training 1
    2: BLUE,     # Training 2
    3: RED,      # Task
}

PROTOCOL_LABELS = {
    1: "Training 1",
    2: "Training 2",
    3: "Task"
}


# =============================================================================
# BASIC HELPERS
# =============================================================================
def ensure_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return []
    return []


def smooth_discrete_curve_fixed_range(x_vals, y_vals, x_min=1, x_max=25, sigma=1.0, points=900):
    x_vals = np.asarray(x_vals, dtype=float)
    y_vals = np.asarray(y_vals, dtype=float)

    x_grid = np.linspace(x_min, x_max, points)
    y_smooth = np.zeros_like(x_grid)

    for xi, yi in zip(x_vals, y_vals):
        y_smooth += yi * np.exp(-0.5 * ((x_grid - xi) / sigma) ** 2)

    dx = x_grid[1] - x_grid[0]
    area = y_smooth.sum() * dx

    if area > 0:
        y_smooth /= area

    return x_grid, y_smooth


def shade_color(base_rgb, p):
    if p is None or pd.isna(p):
        return (*base_rgb, 0.42)

    x = p / 0.30
    r, g, b = base_rgb
    h, l, s = colorsys.rgb_to_hls(r, g, b)

    new_l = 0.96 - 0.14 * x
    new_s = min(1, s + (0.12 + 0.15 * x))
    nr, ng, nb = colorsys.hls_to_rgb(h, new_l, new_s)
    return (nr, ng, nb, 0.42)


def parse_proba(p):
    if p is None or (isinstance(p, float) and pd.isna(p)):
        return None
    p = str(p)
    if "/" not in p:
        return None
    try:
        return float(p.split("/")[1])
    except Exception:
        return None


def protocol_name(proto):
    try:
        return PROTOCOL_LABELS.get(int(proto), f"Protocol {int(proto)}")
    except Exception:
        return "-"


def get_protocol_blocks(df_session):
    blocks = []
    start = None
    curr_p = None
    curr_proto = None
    for _, r in df_session.iterrows():
        s = r["SessionIndex"]
        p = r["Proba_val"]
        proto = r["Protocol"]
        if start is None:
            start, curr_p, curr_proto = s, p, proto
            continue
        if p != curr_p or proto != curr_proto:
            blocks.append((start, s - 1, curr_p, curr_proto))
            start, curr_p, curr_proto = s, p, proto
    if start is not None:
        blocks.append((start, df_session["SessionIndex"].max(), curr_p, curr_proto))
    return blocks


# =============================================================================
# DATA PREP
# =============================================================================
def prepare_data(path):
    df = pd.read_feather(path).copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Date_norm"] = df["Date"].dt.normalize()
    df["Mouse_ID"] = df["Mouse_ID"].astype(str)
    df["Version"] = df["Version"].astype(str)
    df["Protocol"] = pd.to_numeric(df["Protocol"], errors="coerce").astype("Int64")
    df["Proba_val"] = df["Probas"].apply(parse_proba)
    df = df.sort_values(["Mouse_ID", "Date"])

    session_cmap = LinearSegmentedColormap.from_list(
        "session_cmap",
        [PROTOCOL_HEX[2], PROTOCOL_HEX[3]]
    )
    return df, session_cmap


# =============================================================================
# CORE ANALYSIS
# =============================================================================
def compute_failures(row):
    timestamps = row["Timestamps"]
    bouts = row["Bout for Timestamps"]
    if not isinstance(timestamps, np.ndarray) or len(timestamps) == 0:
        return []
    if not isinstance(bouts, np.ndarray) or len(bouts) == 0:
        return []

    rewarded = []
    rw = row.get("Times Rewarded Licks", [])
    if isinstance(rw, np.ndarray) and len(rw) > 0:
        if isinstance(rw[0], (list, np.ndarray)):
            for block in rw:
                rewarded.extend(block.tolist())
        else:
            rewarded = rw.tolist()

    non_rewarded = []
    nra = row.get("Times Non Rewarded Licks", [])
    if isinstance(nra, np.ndarray) and len(nra) > 0:
        if isinstance(nra[0], (list, np.ndarray)):
            for block in nra:
                non_rewarded.extend(block.tolist())
        else:
            non_rewarded = nra.tolist()

    failures = []
    for b in np.unique(bouts):
        mask = bouts == b
        t_in = timestamps[mask]
        if len(t_in) == 0:
            failures.append(0)
            continue
        r_in = [t for t in rewarded if t_in[0] <= t <= t_in[-1]]
        nr_in = [t for t in non_rewarded if t_in[0] <= t <= t_in[-1]]
        if len(nr_in) == 0:
            failures.append(0)
            continue
        if len(r_in) == 0:
            failures.append(len(nr_in))
        else:
            last_r = max(r_in)
            count = sum(1 for t in nr_in if t > last_r)
            failures.append(count)
    return failures


def count_reward_per_bout(row):
    timestamps = row.get("Timestamps")
    bouts = row.get("Bout for Timestamps")
    if not isinstance(timestamps, np.ndarray) or len(timestamps) == 0:
        return []

    rewarded = []
    rw = row.get("Times Rewarded Licks", [])
    if isinstance(rw, np.ndarray) and len(rw) > 0:
        if isinstance(rw[0], (list, np.ndarray)):
            for block in rw:
                rewarded.extend(block if not isinstance(block, np.ndarray) else block.tolist())
        else:
            rewarded = rw.tolist()

    counts = []
    for b in np.unique(bouts):
        mask = bouts == b
        t_in = timestamps[mask]
        if len(t_in) == 0:
            counts.append(0)
            continue
        r_in = [t for t in rewarded if t_in[0] <= t <= t_in[-1]]
        counts.append(len(r_in))
    return counts


def count_licks(row, lick_type):
    column_map = {
        "rewarded": "Times Rewarded Licks",
        "non_rewarded": "Times Non Rewarded Licks",
        "invalid": "Times Invalid Licks"
    }
    column_name = column_map.get(lick_type)
    if not column_name:
        return 0

    data = row.get(column_name)
    if not isinstance(data, np.ndarray) or len(data) == 0:
        return 0

    if len(data) > 0 and isinstance(data[0], (list, np.ndarray)):
        return sum(len(x) for x in data)
    return len(data)


# =============================================================================
# OVERVIEW PLOTS
# =============================================================================
def plot_protocol_strip(df, mouse):
    df_mouse = df[df["Mouse_ID"] == mouse].copy()
    if df_mouse.empty:
        return None

    counts = (
        df_mouse["Protocol"]
        .dropna()
        .astype(int)
        .value_counts()
        .reindex([1, 2, 3], fill_value=0)
    )

    total = counts.sum()
    if total == 0:
        return None

    fig, ax = plt.subplots(figsize=(14, 1.25))
    left = 0

    for proto in [1, 2, 3]:
        width = counts[proto]
        if width <= 0:
            continue

        ax.barh([0], [width], left=left, color=PROTOCOL_HEX[proto], height=0.32)

        center = left + width / 2
        ax.text(
            center,
            0,
            f"{PROTOCOL_LABELS[proto]} ({int(width)})",
            ha="center",
            va="center",
            fontsize=9.5,
            color=COLORS["navy"]
        )
        left += width

    ax.set_xlim(0, total)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title(f"{mouse} — Session types", pad=8, color=COLORS["navy"])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    return fig


def plot_bout_count_rewards(df, mouse):
    df_session = df[df["Mouse_ID"] == mouse].copy()
    if df_session.empty:
        return None

    df_session = df_session.sort_values("Date").reset_index(drop=True)
    df_session["SessionIndex"] = range(1, len(df_session) + 1)

    fig, ax = plt.subplots(figsize=(14, 5.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for _, row in df_session.iterrows():
        s = row["SessionIndex"]
        proto = row["Protocol"]
        p = row["Proba_val"]
        base = PROTOCOL_COLORS.get(proto, (0.93, 0.93, 0.93))
        ax.axvspan(s - 0.5, s + 0.5, color=shade_color(base, p), zorder=1)

    blocks = get_protocol_blocks(df_session)

    ax.plot(
        df_session["SessionIndex"],
        df_session["Number of Bouts"],
        marker="o",
        markersize=4.5,
        linewidth=2.0,
        color=COLORS["navy"],
        zorder=20,
        label="Bouts"
    )

    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin

    df_session["RewardedCount"] = df_session["Number of Rewarded Licks"].fillna(0)
    ax.plot(
        df_session["SessionIndex"],
        df_session["RewardedCount"],
        marker="s",
        markersize=4.2,
        linestyle="--",
        linewidth=1.9,
        color=COLORS["green"],
        zorder=25,
        label="Rewarded licks"
    )

    max_reward = df_session["RewardedCount"].max()
    new_ymax = max(ymax, max_reward * 1.12 if pd.notna(max_reward) else ymax)
    ax.set_ylim(ymin - yrange * 0.12, new_ymax)

    for _, row in df_session.iterrows():
        ax.text(
            row["SessionIndex"],
            row["RewardedCount"] + new_ymax * 0.016,
            str(int(row["RewardedCount"])),
            ha="center",
            fontsize=8,
            color=COLORS["green"],
            zorder=50
        )

    for i, (start, end, p, proto) in enumerate(blocks):
        if p is not None:
            center = (start + end) / 2
            offset = 0.04 if i % 2 == 0 else 0.08
            ax.text(
                center,
                ymin - yrange * offset,
                f"{p:.2f}",
                ha="center",
                va="top",
                fontsize=8.5,
                color=COLORS["gray"]
            )

    ax.set_xlabel("Training sessions", color=COLORS["navy"])
    ax.set_ylabel("Counts", color=COLORS["navy"])
    ax.set_title(f"Mouse {mouse} — Bout count + rewarded licks", color=COLORS["navy"])
    ax.set_xticks(df_session["SessionIndex"])
    ax.grid(alpha=0.22, axis="y", color=COLORS["grid"])

    ax.legend(handles=[
        mpatches.Patch(color=PROTOCOL_HEX[1], label="Training 1"),
        mpatches.Patch(color=PROTOCOL_HEX[2], label="Training 2"),
        mpatches.Patch(color=PROTOCOL_HEX[3], label="Task"),
        mlines.Line2D([], [], color=COLORS["navy"], marker="o", label="Bouts"),
        mlines.Line2D([], [], color=COLORS["green"], marker="s", linestyle="--", label="Rewarded licks"),
    ], loc="upper left", frameon=False)

    fig.tight_layout()
    return fig


def plot_stacked_lick_counts(df, mouse):
    df_session = df[df["Mouse_ID"] == mouse].copy()
    if df_session.empty:
        return None

    df_session = df_session.sort_values("Date").reset_index(drop=True)
    df_session["SessionIndex"] = range(1, len(df_session) + 1)

    reward_total = [count_licks(row, "rewarded") for _, row in df_session.iterrows()]
    nonreward_total = [count_licks(row, "non_rewarded") for _, row in df_session.iterrows()]
    invalid_total = [count_licks(row, "invalid") for _, row in df_session.iterrows()]

    fig, ax = plt.subplots(figsize=(14, 5.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for _, row in df_session.iterrows():
        s = row["SessionIndex"]
        proto = row["Protocol"]
        p = row["Proba_val"]
        base = PROTOCOL_COLORS.get(proto, (0.93, 0.93, 0.93))
        ax.axvspan(s - 0.5, s + 0.5, color=shade_color(base, p), zorder=1)

    blocks = get_protocol_blocks(df_session)

    x = df_session["SessionIndex"].to_numpy()
    ax.bar(x, reward_total, width=0.62, color=YELLOW, label="Rewarded", zorder=40)
    ax.bar(x, nonreward_total, bottom=reward_total, width=0.62, color=RED, label="Non-rewarded", zorder=40)
    ax.bar(
        x,
        invalid_total,
        bottom=np.array(reward_total) + np.array(nonreward_total),
        width=0.62,
        color=BLUE,
        label="Invalid",
        zorder=40
    )

    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin

    for i, (start, end, p, proto) in enumerate(blocks):
        if p is not None:
            center = (start + end) / 2
            offset = 0.04 if i % 2 == 0 else 0.08
            ax.text(
                center,
                ymin - yrange * offset,
                f"{p:.2f}",
                ha="center",
                va="top",
                fontsize=8.5,
                color=COLORS["gray"]
            )

    ax.set_ylim(ymin - yrange * 0.12, ymax)
    ax.set_title(f"Mouse {mouse} — Lick counts per session", color=COLORS["navy"])
    ax.set_xlabel("Training sessions", color=COLORS["navy"])
    ax.set_ylabel("Total licks", color=COLORS["navy"])
    ax.set_xticks(x)
    ax.grid(alpha=0.22, axis="y", color=COLORS["grid"])

    lick_legend = [
        mpatches.Patch(color=YELLOW, label="Rewarded"),
        mpatches.Patch(color=RED, label="Non-rewarded"),
        mpatches.Patch(color=BLUE, label="Invalid")
    ]
    protocol_legend = [
        mpatches.Patch(color=PROTOCOL_HEX[1], label="Training 1"),
        mpatches.Patch(color=PROTOCOL_HEX[2], label="Training 2"),
        mpatches.Patch(color=PROTOCOL_HEX[3], label="Task")
    ]
    ax.legend(handles=lick_legend + protocol_legend, loc="upper left", frameon=False)

    fig.tight_layout()
    return fig


def plot_histogram_kde_failures(df, mouse):
    df_session = df[df["Mouse_ID"] == mouse].copy()
    if df_session.empty:
        return None

    df_session["ConsFailures"] = df_session.apply(compute_failures, axis=1)
    all_failures = np.array([v for L in df_session["ConsFailures"] for v in L])

    if len(all_failures) == 0:
        return None

    fig, ax = plt.subplots(figsize=(8, 5.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    max_x = min(50, int(all_failures.max())) if len(all_failures) else 50
    max_x = max(max_x, 5)

    ax.hist(
        all_failures,
        bins=30,
        range=(0, max_x),
        density=True,
        color=BLUE,
        alpha=0.35,
        edgecolor=BLUE,
        linewidth=1.0
    )

    if len(np.unique(all_failures)) > 1:
        kde = gaussian_kde(all_failures, bw_method=0.08)
        xs = np.linspace(0, max_x, 300)
        ys = kde(xs)
        ax.plot(xs, ys, color=BLUE, lw=2.3)

    ax.set_title(f"Mouse {mouse} — Distribution of consecutive failures", color=COLORS["navy"])
    ax.set_xlabel("Consecutive failures", color=COLORS["navy"])
    ax.set_ylabel("Density", color=COLORS["navy"])
    ax.set_xlim(0, max_x)
    ax.grid(alpha=0.22, axis="y", color=COLORS["grid"])

    fig.tight_layout()
    return fig


def plot_kde_failures_by_session(df, mouse, session_cmap, bandwidth_factor=0.8):
    df_session = df[
        (df["Mouse_ID"] == mouse) &
        (df["Protocol"] == 3) &
        (df["Proba_val"] == 0.30)
    ].copy()

    if df_session.empty:
        return None

    df_session = df_session.sort_values("Date").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 5.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    valid_sessions = []
    for _, row in df_session.iterrows():
        failures = np.array(compute_failures(row), dtype=float)
        valid_bouts = np.asarray(row["Correct Bouts"], dtype=bool)

        if len(valid_bouts) == len(failures):
            failures = failures[valid_bouts]

        failures = failures[np.isfinite(failures)]
        failures = failures[failures > 0]

        if len(failures) < 100:
            continue

        valid_sessions.append((row["Date"], failures))

    if len(valid_sessions) == 0:
        plt.close(fig)
        return None

    xs = np.linspace(0, 30, 400)
    n_sessions = len(valid_sessions)

    for idx, (date_val, failures) in enumerate(valid_sessions):
        kde = gaussian_kde(failures)
        kde.set_bandwidth(kde.factor * bandwidth_factor)
        ys = kde(xs)
        color = session_cmap(idx / max(1, n_sessions - 1))
        ax.plot(xs, ys, linewidth=2.1, color=color, label=f"{date_val.strftime('%Y-%m-%d')}  (n={len(failures)})")

    ax.set_title(f"Mouse {mouse} — KDE of consecutive failures by task session", color=COLORS["navy"])
    ax.set_xlabel("Consecutive failures", color=COLORS["navy"])
    ax.set_ylabel("Density", color=COLORS["navy"])
    ax.set_xlim(0, 30)
    ax.set_xticks(np.arange(0, 31, 5))
    ax.grid(alpha=0.22, axis="y", color=COLORS["grid"])
    ax.legend(title="Session", fontsize=8, frameon=False)

    fig.tight_layout()
    return fig


def plot_regression_rewards_failures_and_slope(
    df,
    mouse,
    session_cmap,
    max_reward=7,
    max_failure=30,
    min_valid_bouts=100
):
    df_session = df[
        (df["Mouse_ID"] == mouse) &
        (df["Protocol"] == 3) &
        (df["Proba_val"] == 0.30)
    ].copy()

    if df_session.empty:
        return None

    df_session = df_session.sort_values("Date").reset_index(drop=True)
    valid_sessions = []

    for _, row in df_session.iterrows():
        failures = np.asarray(compute_failures(row), dtype=float)
        rewards = np.asarray(count_reward_per_bout(row), dtype=float)
        valid_bouts = np.asarray(row["Correct Bouts"], dtype=bool)

        n = min(len(failures), len(rewards), len(valid_bouts))
        failures = failures[:n]
        rewards = rewards[:n]
        valid_bouts = valid_bouts[:n]

        failures = failures[valid_bouts]
        rewards = rewards[valid_bouts]

        mask = np.isfinite(failures) & np.isfinite(rewards)
        failures = failures[mask]
        rewards = rewards[mask]

        mask = failures > 0
        failures = failures[mask]
        rewards = rewards[mask]

        n_valid_base = len(failures)
        if n_valid_base < min_valid_bouts:
            continue

        mask = rewards <= max_reward
        failures_cut = failures[mask]
        rewards_cut = rewards[mask]

        mask = failures_cut <= max_failure
        failures_cut = failures_cut[mask]
        rewards_cut = rewards_cut[mask]

        if len(failures_cut) < 2 or len(np.unique(rewards_cut)) < 2:
            continue

        model = LinearRegression()
        model.fit(rewards_cut.reshape(-1, 1), failures_cut)

        valid_sessions.append({
            "date": pd.to_datetime(row["Date"]),
            "n_base": n_valid_base,
            "n_cut": len(failures_cut),
            "slope": float(model.coef_[0]),
            "intercept": float(model.intercept_),
            "mean_failures": float(np.mean(failures_cut))
        })

    if len(valid_sessions) == 0:
        return None

    fig, axs = plt.subplots(1, 3, figsize=(20, 5.2), gridspec_kw={"wspace": 0.30})
    fig.patch.set_facecolor("white")
    ax_left, ax_mid, ax_right = axs

    n_sessions = len(valid_sessions)
    x_line = np.linspace(1, max_reward, 200)

    for idx, sess in enumerate(valid_sessions):
        color = session_cmap(idx / max(1, n_sessions - 1))
        y_line = sess["slope"] * x_line + sess["intercept"]
        ax_left.plot(
            x_line,
            y_line,
            color=color,
            linewidth=2.0,
            label=f"{sess['date'].strftime('%Y-%m-%d')} (n={sess['n_base']}, cut={sess['n_cut']})"
        )

    ax_left.set_title(f"Mouse {mouse} — Reward vs failures", color=COLORS["navy"])
    ax_left.set_xlabel("Reward count", color=COLORS["navy"])
    ax_left.set_ylabel("Consecutive failures", color=COLORS["navy"])
    ax_left.set_xlim(1, max_reward)
    ax_left.set_ylim(1, max_failure)
    ax_left.set_xticks(np.arange(1, max_reward + 1))
    ax_left.grid(alpha=0.22, axis="y", color=COLORS["grid"])
    ax_left.legend(title="Session", fontsize=8, frameon=False)

    dates = [sess["date"] for sess in valid_sessions]
    slopes = np.array([sess["slope"] for sess in valid_sessions], dtype=float)
    x_idx = np.arange(len(valid_sessions))

    ax_mid.plot(x_idx, slopes, color=BLUE, linewidth=2.0, marker="o", markersize=4.5)
    if len(slopes) >= 2:
        trend_model = LinearRegression()
        trend_model.fit(x_idx.reshape(-1, 1), slopes)
        x_fit = np.linspace(x_idx.min(), x_idx.max(), 200)
        y_fit = trend_model.predict(x_fit.reshape(-1, 1))
        ax_mid.plot(x_fit, y_fit, linestyle="--", linewidth=1.9, color=RED)

    ax_mid.set_title("Slope over time", color=COLORS["navy"])
    ax_mid.set_xlabel("Session date", color=COLORS["navy"])
    ax_mid.set_ylabel("Linear slope", color=COLORS["navy"])
    ax_mid.set_xticks(x_idx)
    ax_mid.set_xticklabels([d.strftime("%Y-%m-%d") for d in dates], rotation=45, ha="right")
    ax_mid.grid(alpha=0.22, axis="y", color=COLORS["grid"])

    mean_failures = np.array([sess["mean_failures"] for sess in valid_sessions], dtype=float)
    ax_right.plot(x_idx, mean_failures, color=YELLOW, linewidth=2.0, marker="o", markersize=4.5)
    if len(mean_failures) >= 2:
        trend_model = LinearRegression()
        trend_model.fit(x_idx.reshape(-1, 1), mean_failures)
        x_fit = np.linspace(x_idx.min(), x_idx.max(), 200)
        y_fit = trend_model.predict(x_fit.reshape(-1, 1))
        ax_right.plot(x_fit, y_fit, linestyle="--", linewidth=1.9, color=RED)

    ax_right.set_title("Mean failures over time", color=COLORS["navy"])
    ax_right.set_xlabel("Session date", color=COLORS["navy"])
    ax_right.set_ylabel("Mean consecutive failures", color=COLORS["navy"])
    ax_right.set_xticks(x_idx)
    ax_right.set_xticklabels([d.strftime("%Y-%m-%d") for d in dates], rotation=45, ha="right")
    ax_right.grid(alpha=0.22, axis="y", color=COLORS["grid"])

    fig.tight_layout()
    return fig


# =============================================================================
# SESSION FOCUS PLOTS
# =============================================================================
def prepare_session_arrays(session, reward_cut=7):
    rewards = np.asarray(ensure_list(session["Rewards"]), dtype=float)
    failures = np.asarray(ensure_list(session["Licks After"]), dtype=float)
    valid_bouts = np.asarray(session["Correct Bouts"], dtype=bool)

    n = min(len(rewards), len(failures), len(valid_bouts))
    rewards = rewards[:n]
    failures = failures[:n]
    valid_bouts = valid_bouts[:n]

    rewards = rewards[valid_bouts]
    failures = failures[valid_bouts]

    mask = np.isfinite(rewards) & np.isfinite(failures)
    rewards = rewards[mask]
    failures = failures[mask]

    mask = failures > 0
    rewards = rewards[mask]
    failures = failures[mask]

    mask = rewards <= reward_cut
    rewards = rewards[mask]
    failures = failures[mask]

    return rewards, failures, int(np.sum(valid_bouts))


def build_session_plot_rewards_vs_failures(session, mouse_id, date_str, reward_cut=7):
    rewards, failures, _ = prepare_session_arrays(session, reward_cut=reward_cut)

    title_suffix = f" - {session['Probas']}" if "Probas" in session.index else ""
    title = f"{mouse_id} - {date_str}{title_suffix}"

    fig, ax = plt.subplots(figsize=(6.6, 4.9))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    if len(rewards) > 0:
        reward_bins = np.sort(np.unique(rewards.astype(int)))
        means = np.array([failures[rewards == r].mean() for r in reward_bins])
        stds = np.array([failures[rewards == r].std() for r in reward_bins])
        counts = np.array([np.sum(rewards == r) for r in reward_bins])

        ax.fill_between(reward_bins, means - stds, means + stds, color=BLUE, alpha=0.25)
        ax.plot(reward_bins, means, color=BLUE, linewidth=2.3, marker="o", markersize=4.5)

        for x, y, c in zip(reward_bins, means, counts):
            ax.text(x, y + 0.25, str(int(c)), ha="center", va="bottom", fontsize=8, color=COLORS["gray"])

    ax.set_title(title, color=COLORS["navy"])
    ax.set_xlabel("Consecutive rewards", color=COLORS["navy"])
    ax.set_ylabel("Consecutive failures", color=COLORS["navy"])
    ax.set_xlim(0.8, reward_cut + 0.2)
    ax.set_ylim(bottom=1)
    ax.set_xticks(np.arange(1, reward_cut + 1))
    ax.grid(axis="y", alpha=0.22, color=COLORS["grid"])

    fig.tight_layout()
    return fig


def build_session_plot_failure_distribution(session, failure_xlim=(0, 25), reward_cut=7):
    rewards, failures, _ = prepare_session_arrays(session, reward_cut=reward_cut)
    _ = rewards

    fig, ax = plt.subplots(figsize=(6.6, 4.9))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    failures_dist = failures.astype(int) if len(failures) > 0 else np.array([], dtype=int)
    failures_dist = failures_dist[(failures_dist >= 1) & (failures_dist <= failure_xlim[1])]

    all_x = np.arange(1, failure_xlim[1] + 1)
    counts_full = np.zeros_like(all_x, dtype=float)

    if len(failures_dist) > 0:
        vals, cnts = np.unique(failures_dist, return_counts=True)
        for v, c in zip(vals, cnts):
            if v in all_x:
                counts_full[all_x == v] = c

    p_emp = counts_full / counts_full.sum() if counts_full.sum() > 0 else counts_full

    ax.bar(all_x, p_emp, width=0.8, color=RED, edgecolor=RED, alpha=0.35, linewidth=1.0)

    x_smooth, y_smooth = smooth_discrete_curve_fixed_range(
        all_x, p_emp, x_min=1, x_max=failure_xlim[1], sigma=1.0, points=900
    )

    ax.plot(x_smooth, y_smooth, color=RED, linewidth=2.4)
    ax.fill_between(x_smooth, y_smooth, color=RED, alpha=0.18)

    ax.set_xlabel("Consecutive failures", color=COLORS["navy"])
    ax.set_ylabel("Probability", color=COLORS["navy"])
    ax.set_xlim(1, failure_xlim[1])
    ax.set_xticks(np.arange(1, failure_xlim[1] + 1, 5))
    ax.grid(axis="y", alpha=0.22, color=COLORS["grid"])

    fig.tight_layout()
    return fig


# =============================================================================
# CALENDAR HELPERS
# =============================================================================
def get_month_options(valid_dates):
    return sorted({(d.year, d.month) for d in valid_dates})


def month_to_label(year, month):
    return f"{year}-{month:02d}"


def build_month_day_list(year, month):
    return [d for d in calendar.Calendar(firstweekday=0).itermonthdates(year, month) if d.month == month]


def get_session_date_to_protocol(df_sessions):
    out = {}
    for _, row in df_sessions.iterrows():
        d = pd.Timestamp(row["Date_norm"]).date()
        proto = row["Protocol"]
        out[d] = int(proto) if pd.notna(proto) else None
    return out