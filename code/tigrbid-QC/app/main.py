# %%

import math
import os
from datetime import datetime
from glob import glob
from pathlib import Path

import pandas as pd
import streamlit as st
from models import MetricQC, QCRecord
from PIL import Image
from streamlit_scroll_to_top import scroll_to_here

for k, v in st.session_state.items():
    st.session_state[k] = v

freesurfer_metrics = pd.read_csv(
    "/projects/ttan/tmp_test/SCanD_project/data/local/derivatives/freesurfer/7.4.1/00_group2_stats_tables/euler.tsv",
    sep="\t",
)

fmriprep_derivative = (
    "/projects/ttan/tmp_test/SCanD_project/data/local/derivatives/fmriprep/23.2.3"
)

reconall_svgs = sorted(
    glob(f"{fmriprep_derivative}/sub-*/figures/sub-*_desc-reconall_T1w.svg")
)


def scroll():
    st.session_state.scroll_to_top = True


def scrollheader():
    st.session_state.scroll_to_header = True


st.title("Freesurfer QC")
rater_name = st.text_input("Rater name:")
# Show the value dynamically
st.write("You entered:", rater_name)
rater_name = "Thomas"

# Session State Initialization
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

# Calculate current batch ONCE with correct indices
start_idx = (st.session_state.current_page - 1) * st.session_state.batch_size
end_idx = min(start_idx + st.session_state.batch_size, len(reconall_svgs))
# Ensure we don't exceed list length
current_batch = reconall_svgs[start_idx:end_idx]

# Debug info (remove in production)
st.write(
    f"You currently on : Page {st.session_state.current_page}, Batch size {st.session_state.batch_size}, Start {start_idx}, End {end_idx}, Total images {len(reconall_svgs)}"
)
svg_path = current_batch[0]
# Go through the batch of images
for svg_path in current_batch:
    base_svg = os.path.basename(svg_path)
    parts = base_svg.split("_")
    sub_id = parts[0].split("-")[1]
    ses_id = next((p for p in parts if p.startswith("ses-")), None)
    if ses_id:
        ses_num = ses_id.split("-")[1]
    else:
        ses_id = "ses-01"
        ses_num = ses_id.split("-")[1]
    run_id = next((p for p in parts if p.startswith("run-")), None)
    log_file = Path(
        f"{fmriprep_derivative}/sourcedata/freesurfer/sub-{sub_id}/scripts/recon-all-status.log"
    )
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
            formatted_date = complete_time.strftime("%m-%d-%Y %H:%M:%S.%f")
    else:
        complete_time = None
        st.error(f"Log file not found for subject {sub_id}.")

    st.header(f"Subject {sub_id} Session {ses_num}")

    # Extract Euler metric (numeric)
    filtered = freesurfer_metrics[
        (freesurfer_metrics["subject"].str.contains(f"sub-{sub_id}"))
        & (freesurfer_metrics["subject"].str.contains(f"{ses_id}"))
    ]
    row = filtered.squeeze() if not filtered.empty else None
    l_euler_val = row.get("lh_euler") if row is not None else None
    r_euler_val = row.get("rh_euler") if row is not None else None
    euler_vals = {"Left": l_euler_val, "Right": r_euler_val}

    # Clear metrics per subject
    subject_metrics = []

    for hemi, val in euler_vals.items():
        euler_key = f"{sub_id}_euler_{hemi}"
        st.markdown(f"<h4>{hemi} Euler value: {val}</h4>", unsafe_allow_html=True)
        st.radio(
            "",
            options=("PASS", "FAIL", "UNCERTAIN"),
            key=euler_key,
            label_visibility="collapsed",
            index=None,
        )
        qc_choice = st.session_state.get(euler_key)
        metric = MetricQC(name=f"Euler_{hemi}", value=val, qc=qc_choice)

        # Avoid duplicating entries on rerun
        if not any(
            m.name == metric.name and m.value == metric.value
            for m in st.session_state.metrics
        ):
            subject_metrics.append(metric)

    # Segmentation SVG (visual)
    st.image(svg_path, width="stretch")
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

    # Other manual fields
    require_rerun = st.radio(
        f"Require rerun?", ("YES", "NO"), key=f"{sub_id}_rerun", index=None
    )

    if require_rerun is None:
        final_qc = None
    else:
        final_qc = "FAIL" if require_rerun == "YES" else "PASS"

    notes = st.text_input(f"***NOTES***", key=f"{sub_id}_notes")

    # Create a metric for notes
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
total_pages = max(1, math.ceil(len(reconall_svgs) / st.session_state.batch_size))

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

    if col3.button("➡️"):
        if st.session_state.current_page < total_pages:
            st.session_state.current_page += 1
            st.rerun()  # Force rerun to update immediately

with bottom_menu[0]:
    st.markdown(f"Page **{st.session_state.current_page}** of **{total_pages}**")

st.button("Scroll to Top", on_click=scroll)

# Save to CSV
if st.button("Save QC results to CSV"):
    out_file = Path("/projects/ttan/tmp_test/qc_results.csv")
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
                "notes": next(
                    (m.notes for m in rec.metrics if m.name == "QC_notes"), None
                ),
            }
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    pd.DataFrame(rows).to_csv(out_file, index=False)
    st.success(f"QC results saved to {out_file}")
