"""Static visual system for the Streamlit Research Desk."""

from __future__ import annotations

import streamlit as st

APP_CSS = """
<style>
:root {
  --desk-canvas: #f6f6f2;
  --desk-surface: #ffffff;
  --desk-ink: #17211f;
  --desk-muted: #65706d;
  --desk-line: #dfe4e1;
  --desk-accent: #0f766e;
  --desk-accent-hover: #0b5f59;
  --desk-accent-soft: #e8f4f1;
  --desk-danger: #b42318;
  --desk-radius: 12px;
  --desk-shadow: 0 1px 2px rgba(23, 33, 31, 0.05);
}

[data-testid="stAppViewContainer"] {
  background: var(--desk-canvas);
  color: var(--desk-ink);
}

[data-testid="stHeader"] {
  background: color-mix(in srgb, var(--desk-canvas) 88%, transparent);
}

[data-testid="stSidebar"] {
  background: #f0f1ed;
  border-right: 1px solid var(--desk-line);
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] label {
  color: var(--desk-ink);
}

.block-container {
  max-width: 1120px;
  padding-top: 2.25rem;
  padding-bottom: 4rem;
}

h1, h2, h3, h4 {
  color: var(--desk-ink);
  letter-spacing: -0.02em;
}

h1 {
  font-size: clamp(2rem, 3vw, 2.75rem);
  line-height: 1.08;
}

p, li, label, [data-testid="stCaptionContainer"] {
  line-height: 1.55;
}

[data-testid="stForm"],
[data-testid="stExpander"],
[data-testid="stMetric"],
[data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--desk-surface);
  border-color: var(--desk-line);
  border-radius: var(--desk-radius);
  box-shadow: var(--desk-shadow);
}

[data-testid="stForm"] {
  padding: 1.2rem 1.25rem 0.4rem;
}

.stButton > button,
.stDownloadButton > button,
[data-testid="stFormSubmitButton"] > button,
[data-baseweb="select"] > div,
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stFileUploaderDropzone"] button {
  min-height: 44px;
  border-radius: 9px;
}

.stButton > button[kind="primary"],
[data-testid="stFormSubmitButton"] > button[kind="primary"] {
  background: var(--desk-accent);
  border-color: var(--desk-accent);
  color: #ffffff;
  font-weight: 650;
}

.stButton > button[kind="primary"]:hover,
[data-testid="stFormSubmitButton"] > button[kind="primary"]:hover {
  background: var(--desk-accent-hover);
  border-color: var(--desk-accent-hover);
}

button:focus-visible,
input:focus-visible,
textarea:focus-visible,
[role="combobox"]:focus-visible,
[role="radio"]:focus-visible,
a:focus-visible {
  outline: 3px solid rgba(15, 118, 110, 0.34) !important;
  outline-offset: 2px !important;
}

[data-testid="stAlert"] {
  border-radius: 10px;
  border: 1px solid var(--desk-line);
}

[data-testid="stTabs"] [data-baseweb="tab-list"] {
  gap: 0.25rem;
  border-bottom: 1px solid var(--desk-line);
}

[data-testid="stTabs"] [data-baseweb="tab"] {
  min-height: 44px;
}

[data-testid="stProgress"] > div > div {
  background-color: var(--desk-accent);
}

@media (max-width: 760px) {
  .block-container {
    padding: 1.25rem 1rem 3rem;
  }

  [data-testid="column"] {
    min-width: 100% !important;
  }

  [data-testid="stForm"] {
    padding: 1rem 0.9rem 0.3rem;
  }

  [data-testid="stHorizontalBlock"] {
    flex-wrap: wrap;
  }
}

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    scroll-behavior: auto !important;
    transition-duration: 0.01ms !important;
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
  }
}
</style>
"""


def inject_theme() -> None:
    """Apply the static app theme without interpolating runtime content."""

    st.markdown(APP_CSS, unsafe_allow_html=True)
