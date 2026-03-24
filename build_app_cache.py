from pathlib import Path
import pandas as pd
import zipfile

from app_functions import (
    prepare_mouse_dataframe,
    plot_protocol_strip,
    plot_bout_count_rewards,
    plot_stacked_lick_counts,
    plot_histogram_kde_failures,
    plot_kde_failures_by_session,
    plot_regression_rewards_failures_and_slope,
    build_session_plot_rewards_vs_failures,
    build_session_plot_failure_distribution,
    save_figure,
)

# =============================================================================
# CONFIG
# =============================================================================
FEATHER_PATH = Path(
    r"\\SynoINVIBE_Caze\INVIBE_team_Cazettes\data\database\full_db_all_rigs.feather"
)

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "app_cache"
OUT_PLOTS = OUT_DIR / "plots"
OUT_META = OUT_DIR / "metadata.parquet"
ZIP_PATH = BASE_DIR / "app_cache.zip"


def safe_name(s: str):
    return str(s).replace("/", "_").replace("\\", "_").replace(" ", "_")


def ensure_plot(fig, path):
    if path.exists():
        return False

    if fig is not None:
        save_figure(fig, path)
        return True

    return False


def build_zip():
    print("\nCréation du ZIP...")
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file in OUT_DIR.rglob("*"):
            if file.is_file():
                zf.write(file, file.relative_to(OUT_DIR))

    print("ZIP créé :", ZIP_PATH)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PLOTS.mkdir(parents=True, exist_ok=True)

    if not FEATHER_PATH.exists():
        raise FileNotFoundError(f"Feather introuvable: {FEATHER_PATH}")

    print("Lecture du feather...")
    df = pd.read_feather(FEATHER_PATH)

    df, session_cmap = prepare_mouse_dataframe(df)
    mice = sorted(df["Mouse_ID"].dropna().astype(str).unique().tolist())

    print("Nb souris :", len(mice))

    meta_rows = []
    total_new_plots = 0

    for mouse_id in mice:
        print(f"\nProcessing {mouse_id}...")
        df_mouse = df[df["Mouse_ID"] == mouse_id].copy()

        if df_mouse.empty:
            continue

        mouse_dir = OUT_PLOTS / safe_name(mouse_id)
        overview_dir = mouse_dir / "overview"
        session_dir = mouse_dir / "sessions"
        overview_dir.mkdir(parents=True, exist_ok=True)
        session_dir.mkdir(parents=True, exist_ok=True)

        # =========================
        # OVERVIEW
        # =========================
        overview_specs = [
            ("protocol_strip", plot_protocol_strip(df_mouse, mouse_id)),
            ("bout_count_rewards", plot_bout_count_rewards(df_mouse, mouse_id)),
            ("stacked_lick_counts", plot_stacked_lick_counts(df_mouse, mouse_id)),
            ("histogram_kde_failures", plot_histogram_kde_failures(df_mouse, mouse_id)),
            ("kde_failures_by_session", plot_kde_failures_by_session(df_mouse, mouse_id, session_cmap)),
            (
                "regression_rewards_failures_and_slope",
                plot_regression_rewards_failures_and_slope(df_mouse, mouse_id, session_cmap),
            ),
        ]

        for name, fig in overview_specs:
            path = overview_dir / f"{name}.png"
            if ensure_plot(fig, path):
                total_new_plots += 1

        # =========================
        # SESSIONS
        # =========================
        for _, row in df_mouse.iterrows():
            date_value = pd.to_datetime(row["Date"])
            date_str = date_value.strftime("%Y-%m-%d")
            version_str = str(row["Version"])
            session_key = f"{date_str}__v{safe_name(version_str)}"

            one_session_dir = session_dir / session_key
            one_session_dir.mkdir(parents=True, exist_ok=True)

            path1 = one_session_dir / "rewards_vs_failures.png"
            path2 = one_session_dir / "failure_distribution.png"

            if not path1.exists():
                fig1 = build_session_plot_rewards_vs_failures(row, mouse_id, date_str)
                if fig1 is not None:
                    save_figure(fig1, path1)
                    total_new_plots += 1

            if not path2.exists():
                fig2 = build_session_plot_failure_distribution(row)
                if fig2 is not None:
                    save_figure(fig2, path2)
                    total_new_plots += 1

            meta_rows.append({
                "Mouse_ID": mouse_id,
                "Date": date_value,
                "Date_norm": date_value.normalize(),
                "Version": version_str,
                "Protocol": row["Protocol"],
                "Probas": row["Probas"],
                "Number of Bouts": row.get("Number of Bouts"),
                "Number of Rewarded Licks": row.get("Number of Rewarded Licks"),
                "protocol_strip_path": (overview_dir / "protocol_strip.png").relative_to(OUT_DIR).as_posix(),
                "bout_count_rewards_path": (overview_dir / "bout_count_rewards.png").relative_to(OUT_DIR).as_posix(),
                "stacked_lick_counts_path": (overview_dir / "stacked_lick_counts.png").relative_to(OUT_DIR).as_posix(),
                "histogram_kde_failures_path": (overview_dir / "histogram_kde_failures.png").relative_to(OUT_DIR).as_posix(),
                "kde_failures_by_session_path": (overview_dir / "kde_failures_by_session.png").relative_to(OUT_DIR).as_posix(),
                "regression_rewards_failures_and_slope_path": (
                    (overview_dir / "regression_rewards_failures_and_slope.png").relative_to(OUT_DIR).as_posix()
                ),
                "session_rewards_vs_failures_path": path1.relative_to(OUT_DIR).as_posix(),
                "session_failure_distribution_path": path2.relative_to(OUT_DIR).as_posix(),
            })

    df_meta = pd.DataFrame(meta_rows)

    if not df_meta.empty:
        df_meta = df_meta.sort_values(["Mouse_ID", "Date", "Version"]).reset_index(drop=True)
        df_meta.to_parquet(OUT_META, index=False)

    print("\nNouveaux plots générés :", total_new_plots)

    build_zip()

    print("\nDONE.")
    print("Metadata :", OUT_META)
    print("ZIP :", ZIP_PATH)


if __name__ == "__main__":
    main()