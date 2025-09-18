# %%
import math
import os
import pickle
from argparse import ArgumentParser
from datetime import datetime
from glob import glob
from pathlib import Path

import pandas as pd
import streamlit as st
from models import MetricQC, QCRecord
from session_persistance import load_session_state, save_session_state
from streamlit_scroll_to_top import scroll_to_here


def parse_args(args=None):
    parser = ArgumentParser("Freesurfer QC")

    parser.add_argument(
        "--fs_metric",
        dest="freesurfer_metric",
        help="Group Euler CSV file",
        required=True,
    )

    parser.add_argument(
        "--fmri_dir",
        help=(
            "The root directory of fMRI preprocessing derivatives. "
            "For example, /SCanD_project/data/local/derivatives/fmriprep/23.2.3."
        ),
        required=True,
    )

    parser.add_argument(
        "--output_dir",
        dest="out_dir",
        help="Directory to save session state and QC results",
        required=True,
    )
    return parser.parse_args(args)


args = parse_args()
fs_metric = args.freesurfer_metric
fmri_dir = args.fmri_dir
out_dir = args.out_dir
SESSION_STATE_FILE = Path(out_dir) / "session_state.pkl"

# To `keep` the value when switching page
for k, v in st.session_state.items():
    st.session_state[k] = v

# Read metrics and filter for ses-01
freesurfer_metrics = pd.read_csv(
    fs_metric,
    sep="\t",
)
# Need to adapt the code so it work when there is no ses-01 in the subject column of the dataframe
if freesurfer_metrics["subject"].str.contains("ses-01").any():

    filtered_fs_metrics = freesurfer_metrics[
        freesurfer_metrics["subject"].str.contains("ses-01")
    ]
else:
    filtered_fs_metrics = freesurfer_metrics


def scroll():
    st.session_state.scroll_to_top = True


def scrollheader():
    st.session_state.scroll_to_header = True


st.title("Freesurfer QC")
rater_name = st.text_input("Rater name:")
# Show the value dynamically
st.write("You entered:", rater_name)

# Session State Initialization
if "initialized" not in st.session_state:
    load_session_state(SESSION_STATE_FILE)
    st.session_state.initialized = True

if "metrics" not in st.session_state:
    st.session_state.metrics = []

if "current_page" not in st.session_state:
    st.session_state.current_page = 1

if "batch_size" not in st.session_state:
    st.session_state.batch_size = 5

if "scroll_to_top" not in st.session_state:
    st.session_state.scroll_to_top = False

if "scroll_to_header" not in st.session_state:
    st.session_state.scroll_to_header = False

if st.session_state.scroll_to_top:
    scroll_to_here(0, key="top")  # Scroll to the top of the page
    st.session_state.scroll_to_top = False  # Reset the state after scrolling

if "scroll_to_top" not in st.session_state:
    st.session_state.scroll_to_top = False

if "qc_records" not in st.session_state:
    st.session_state.qc_records = {}

# Pagination
total_rows = len(filtered_fs_metrics)
# Compute batch
start_idx = (st.session_state.current_page - 1) * st.session_state.batch_size
end_idx = min(start_idx + st.session_state.batch_size, total_rows)
current_batch = filtered_fs_metrics.iloc[start_idx:end_idx]

for _, row in current_batch.iterrows():
    subj = row["subject"]
    parts = subj.split("_")
    sub_id = parts[0].split("-")[1]
    ses_id = parts[1] if len(parts) > 1 else "ses-01"
    ses_num = ses_id.split("-")[1] if "-" in ses_id else ses_id
    run_id = None

    # Optional: find reconall SVG
    svg_matches = glob(
        f"{fmri_dir}/sub-{sub_id}/figures/sub-{sub_id}_*desc-reconall_T1w.svg"
    )
    svg_path = svg_matches[0] if svg_matches else None
    # Get recon-all timestamp
    log_file = Path(
        f"{fmri_dir}/sourcedata/freesurfer/sub-{sub_id}/scripts/recon-all-status.log"
    )
    complete_time = None
    if log_file.is_file():
        lines = log_file.read_text().splitlines()
        last_line = lines[-1]
        if "finished without error at" in last_line:
            finished_str = last_line.split(" at ")[1]
            finished_str_no_tz = " ".join(
                finished_str.split()[1:-2] + [finished_str.split()[-1]]
            )
            format_data = "%b %d %H:%M:%S %Y"
            complete_time = datetime.strptime(finished_str_no_tz, format_data)
            formatted_date = complete_time.strftime("%m-%d-%Y")
    else:
        st.warning(f"Log file not found for subject {sub_id}.")

    st.header(f"Subject {sub_id} Session {ses_num}")

    # metrics per subject
    subject_metrics = []

    # Euler metrics
    euler_vals = {"Left": row.get("lh_euler"), "Right": row.get("rh_euler")}
    for hemi, val in euler_vals.items():
        euler_key = f"{sub_id}_euler_{hemi}"
        options = ("PASS", "FAIL", "UNCERTAIN")

        if val < -150 or val > 0:
            default_index = options.index("FAIL")
        elif -150 <= val <= 0:
            default_index = options.index("PASS")
        else:
            default_index = None
        st.markdown(f"<h4>{hemi} Euler value: {val}</h4>", unsafe_allow_html=True)

        st.radio(
            "",
            options=options,
            key=euler_key,
            label_visibility="collapsed",
            index=default_index,
        )

        qc_choice = st.session_state.get(euler_key)
        metric = MetricQC(name=f"Euler_{hemi}", value=val, qc=qc_choice)

        # Avoid duplicating entries on rerun
        if not any(
            m.name == metric.name and m.value == metric.value
            for m in st.session_state.metrics
        ):
            subject_metrics.append(metric)

    # Segmentation SVG
    st.set_page_config(layout="wide")
    if svg_path is not None and os.path.exists(svg_path):
        st.image(svg_path, use_container_width=True)
        # st.image(svg_path, width="stretch")
    else:
        st.warning(f"Image not found: {svg_path}")
    st.markdown(f"<h4> Surface Segmentation QC", unsafe_allow_html=True)
    seg_qc = st.radio(
        "",
        ("PASS", "FAIL", "UNCERTAIN"),
        key=f"{sub_id}_seg",
        label_visibility="collapsed",
        index=None,
    )
    metric = MetricQC(name="surface_segmentation", qc=seg_qc)

    if not any(m.name == metric.name for m in st.session_state.metrics):
        subject_metrics.append(metric)

    # Require rerun
    require_rerun = st.radio(
        f"Require rerun?", ("YES", "NO"), key=f"{sub_id}_rerun", index=None
    )

    if require_rerun is None:
        final_qc = None
    else:
        final_qc = "FAIL" if require_rerun == "YES" else "PASS"

    # Notes
    notes = st.text_input(f"***NOTES***", key=f"{sub_id}_notes")
    subject_metrics.append(MetricQC(name="QC_notes", notes=notes))

    metric_notes = MetricQC(name="QC_notes", notes=notes)
    if not any(m.name == metric_notes.name for m in st.session_state.metrics):
        subject_metrics.append(metric_notes)

    # Create QCRecord
    st.session_state.qc_records[sub_id] = QCRecord(
        subject_id=sub_id,
        session_id=ses_id,
        run_id=run_id,
        pipeline="freesurfer-7.4.1",
        complete_timestamp=formatted_date if "formatted_date" in locals() else None,
        rater=rater_name,
        require_rerun=require_rerun,
        final_qc=final_qc,
        metrics=subject_metrics,
    )

# Pagination Controls - MOVED TO TOP
bottom_menu = st.columns((1, 2, 1))

# Update batch size first
with bottom_menu[2]:
    new_batch_size = st.selectbox(
        "Page Size",
        options=[5, 10, 20],
        index=(
            [5, 10, 20].index(st.session_state.batch_size)
            if st.session_state.batch_size in [5, 10, 20]
            else 0
        ),
    )

    # If batch size changed, reset to page 1
    if new_batch_size != st.session_state.batch_size:
        st.session_state.batch_size = new_batch_size
        st.session_state.current_page = 1
        st.rerun()

# Calculate total pages with current batch size
total_pages = max(1, math.ceil(total_rows / st.session_state.batch_size))
# Navigation controls
with bottom_menu[1]:
    col1, col2, col3 = st.columns([1, 1, 1], gap="small")

    if col1.button("⬅️"):
        if st.session_state.current_page > 1:
            st.session_state.current_page -= 1
            st.rerun()  # Force rerun to update immediately

    new_page = col2.number_input(
        "Page",
        min_value=1,
        max_value=total_pages,
        value=st.session_state.current_page,
        step=1,
    )

    # Update current page if changed
    if new_page != st.session_state.current_page:
        st.session_state.current_page = new_page
        st.rerun()

    if col3.button("➡️"):
        if st.session_state.current_page < total_pages:
            st.session_state.current_page += 1
            st.rerun()  # Force rerun to update immediately

with bottom_menu[0]:
    st.markdown(f"Page **{st.session_state.current_page}** of **{total_pages}**")

st.button("Scroll to Top", on_click=scroll)

# Save to CSV
now = datetime.now()
timestamp = now.strftime("%Y%m%d")  # e.g., 20250917
out_file = Path(out_dir) / f"qc_results_{timestamp}.csv"

if st.button("Save QC results to CSV"):
    save_session_state()
    out_file.parent.mkdir(exist_ok=True, parents=True)

    # Flatten metrics dynamically for CSV
    rows = []
    for rec in st.session_state.qc_records.values():
        row = {
            "subject": f"sub-{rec.subject_id}",
            "session": rec.session_id,
            "run": rec.run_id,
            "pipeline": rec.pipeline,
            "recon_timestamp": rec.complete_timestamp,
        }

        for m in rec.metrics:
            metric_name = m.name.lower().replace("-", "_")
            if m.value is not None:
                row[f"{metric_name}_value"] = m.value
            if m.qc is not None:
                row[f"{metric_name}_qc"] = m.qc

        row.update(
            {
                "require_rerun": rec.require_rerun,
                "rater": rec.rater,
                "final_qc": rec.final_qc,
                "notes": next((m.notes for m in rec.metrics if m.name == "QC_notes")),
            }
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    pd.DataFrame(rows).to_csv(out_file, index=False)

    st.success(f"QC results saved to {out_file}")
