import os
import pickle
from pathlib import Path

import streamlit as st

# QC_FILE = Path("/projects/ttan/tmp_test/qc_records.pkl")
SESSION_STATE_FILE = Path("/projects/ttan/tmp_test/session_state.pkl")


def save_session_state(file_path: Path):
    """
    Save the current Streamlit session state as a pickle file.
    """
    try:
        with open(file_path, "wb") as f:
            pickle.dump(st.session_state.to_dict(), f)
        st.success("Session state saved.")
    except Exception as e:
        st.error(f"Error saving session state: {e}")


def load_session_state(file_path: Path):
    """
    Load a previously saved Streamlit session state pickle file.
    """
    if os.path.exists(file_path):
        try:
            with open(file_path, "rb") as f:
                loaded_state = pickle.load(f)
                for key, value in loaded_state.items():
                    st.session_state[key] = value
            st.info(f"Session state loaded from {file_path}")
        except Exception as e:
            st.error(f"Error loading session state: {e}")
    else:
        st.warning(f"Initialize New Session")
