# -*- coding: utf-8 -*-
"""
Created on Sat Dec 27 08:38:49 2025

@author: Davidg
"""


import streamlit as st
import tempfile
import os

import sys
import random
from math import ceil

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import random
import pandas as pd


from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, landscape
from reportlab.graphics.shapes import Drawing, Rect, Circle, String
from reportlab.graphics import renderPDF
import random
import numpy as np


# ---------------------------------------------------------
#  DRY-FIRE STAGE GENERATOR LOGIC
# ---------------------------------------------------------

POSITIONS = [
    "Standing",
    "Kneeling",
    "Low Kneeling",
    "Squat",
    "Urban Prone (Left)",
    "Urban Prone (Right)",
    "Prone",
    "Supported Barricade (High)",
    "Supported Barricade (Low)",
]


DISTANCE_OPTIONS = [5, 7, 10, 15, 20, 25, 30, 50, 75, 100]

SEQUENCES = [
    "Near → Far",
    "Far → Near",
    "Left → Right",
    "Right → Left",
    "Mixed",
]



# -----------------------------------
# Target Dimensions (inches)
# -----------------------------------
TARGET_TYPES = {
    "plate": {"shape": "circle", "w": 12, "h": 12},
    "ipsc": {"shape": "rect", "w": 18.5, "h": 31},
    "ipsc_1_3": {"shape": "rect", "w": 18.5/3, "h": 31/3},
}


def generate_stage(
    num_targets=3,
    allow_barricade=True,
    allow_prone=True,
    allow_kneeling=True,
    allow_standing=True,
    allow_transitions=True,
    allow_ranging=True,
    use_plates_and_ipsc=True,
    randomize_wind=True
):
    # -----------------------
    # Parameters & Components
    # -----------------------

    target_types = ["IPSC", "Plate"] if use_plates_and_ipsc else ["IPSC"]

    # simulated ranges (yd)
    possible_ranges = [100, 200, 300, 400, 500, 600, 700]
    
    # wind conditions
    wind_conditions = pd.DataFrame([['None','Light', 'Breeze', 'Noticeable', 'Moderate', 'Strong'], 
                                    ['0 mph', '1-3 mph', '4-7 mph', '8-12 mph', '13-18 mph', '20+ mph'], 
                                    ['None', 'Drifting smoke & dust', 'Grass moving', 'Leaves rustling', 'Branches swaying', 'Trees swaying']
                                    ], index=['Description', 'Speed', 'Indicator']).T

    # positions
    selected_positions = []
    positions = []
    if allow_standing:
        positions.append("Standing")
    if allow_kneeling:
        positions.append("Kneeling")
    if allow_prone:
        positions.append("Prone")
    if allow_barricade:
        positions.append("Barricade Supported")

    # Stage sequences
    sequences = ["near-to-far", "far-to-near", "left-to-right", "right-to-left", "mixed"]

    sequence = random.choice(sequences)

    # Select targets
    targets = []
    for i in range(num_targets):
        targets.append({
            "name": f"T{i+1}",
            "type": random.choice(target_types),
            "range": random.choice(possible_ranges)
        })

    # Order targets based on sequence rules
    if sequence == "near-to-far":
        targets.sort(key=lambda x: x["range"])
    elif sequence == "far-to-near":
        targets.sort(key=lambda x: x["range"], reverse=True)
    elif sequence == "left-to-right":
        # just keep T1–TN as ordered but label as L→R
        pass
    elif sequence == "right-to-left":
        targets.reverse()
    else:
        random.shuffle(targets)

    # Pick start and movement
    start_position = random.choice(positions)
    selected_positions.append(start_position)
    movement = None
    if allow_transitions and random.random() < 0.6:
        # 60% chance to include a position transition
        end_position = random.choice([p for p in positions if p != start_position])
        movement = f"Transition from {start_position} to {end_position}"
        selected_positions.append(end_position)
    else:
        end_position = start_position

    # Dial vs hold assignments
    dial_target = [] #min(targets, key=lambda d: d["range"])
    hold_targets = [t for t in targets if t != dial_target]

    # Ranging requirement
    ranging_required = allow_ranging and (random.random() < 0.4)  # 40% chance



    # -----------------------
    # Par Time Calculation
    # -----------------------

    """
    Base time is calculated as:\n
      • 10 sec per position build\n
      • +5 sec if barricade involved\n
      • +5 sec for each additional target after first\n
      • +7 sec if ranging is required\n
      • +10 sec if movement is required
    """

    par_time = 10

    # starting position
    if "Barricade" in start_position:
        par_time += 5

    # transitions
    if movement:
        par_time += 10
        if "Barricade" in end_position:
            par_time += 5

    # targets
    par_time += (num_targets - 1) * 5

    # ranging
    if ranging_required:
        par_time += 7

    # round to clean number
    par_time = int(round(par_time / 5.0) * 5)
    
    if randomize_wind:
        rand_wind = random.randint(0, wind_conditions.shape[0]-1)
        

    # -----------------------
    # Build COF Brief Output
    # -----------------------

    brief = []
    #brief.append("=======================================")
    brief.append(" RANDOM DRY-FIRE MINI STAGE (DMR)\n      ")
    #brief.append("=======================================")
    brief.append("\n")
    brief.append(f"Start Position: {start_position}\n")
    if movement:
        brief.append(f"Movement: {movement}\n")
    else:
        brief.append("Movement: None\n")
    brief.append(f"Target Sequence: {sequence}\n")
    brief.append("\n")

    brief.append("Targets:\n")
    for t in targets:
        brief.append(f"  • {t['name']}: {t['type']} at simulated {t['range']} yd\n")

    brief.append("\n")
    if dial_target: 
        brief.append(f"Dial for: {dial_target['name']} ({dial_target['range']} yd)\n")
    if hold_targets:
        brief.append("Hold for:\n")
        for t in hold_targets:
            brief.append(f"  • {t['name']} ({t['range']} yd)\n")
    brief.append("\n")

    if ranging_required:
        brief.append("Ranging Required: YES (range one called target using reticle)\n")
    else:
        brief.append("Ranging Required: NO\n")

    brief.append("\n")
    brief.append(f"PAR TIME: {par_time} seconds\n")
    brief.append("\n")
    brief.append("Shooter Ready — Standby — *BEEP*\n")
    brief.append("\n")

    if randomize_wind: 
        brief.append("Wind conditions: " + wind_conditions.iloc[rand_wind, 0]+"\n")
        brief.append("Wind observations: " + wind_conditions.iloc[rand_wind, 2]+"\n")
        brief.append("Wind speed: " + wind_conditions.iloc[rand_wind, 1])
        


    return {
        "positions": selected_positions,
        "distances": [t['range'] for t in targets if t in targets],
        "sequence": sequence,
        "target_type": [t['type'] for t in targets if t in targets],
        "par_time": par_time,
        "brief": "\n".join(brief)
    }, targets


# ---------------------------------------------------------
#  PDF EXPORT (SCALED TARGETS)
# ---------------------------------------------------------

# -----------------------------------
# Scaling function
# -----------------------------------
def scaled_inches(value_in, wall_distance, r_sim):
    return value_in * (wall_distance / r_sim)



def build_target_drawing(target_type, wall_distance, sim_distance):
    t = TARGET_TYPES[target_type]

    scaled_w_in = scaled_inches(t["w"], wall_distance, sim_distance)
    scaled_h_in = scaled_inches(t["h"], wall_distance, sim_distance)

    w_pt = scaled_w_in * 72
    h_pt = scaled_h_in * 72

    draw = Drawing(w_pt + 200, h_pt + 40)

    if t["shape"] == "circle":
        r = w_pt / 2
        draw.add(Circle(r, r, r, strokeWidth=2))
    else:
        draw.add(Rect(0, 0, w_pt, h_pt, strokeWidth=2))

    label = f"{target_type.upper()} – Sim: {sim_distance:.0f}"
    draw.add(String(0, h_pt + 10, label, fontSize=12))

    return draw, w_pt, h_pt


def export_pdf(
        file_path,
        target_type="ipsc",
        num_targets=5,
        dist_min=100,
        dist_max=700,
        wall_distance=5,
        page_margin=0.5,  # inches
        targets=None       # NEW: optional list of dicts
    ):

    # ----- NEW: If targets list is provided, use it and ignore other params -----
    if targets is not None:
        # sanitize types and sort by distance descending
        cleaned = []
        for t in targets:
            ttype = t["type"].lower()
            if ttype not in TARGET_TYPES:
                raise ValueError(f"Unknown target type '{t['type']}'. Valid: {list(TARGET_TYPES.keys())}")

            cleaned.append({
                "name": t["name"],
                "type": ttype,
                "range": float(t["range"])
            })

        # Sort by range: largest distance → smallest
        cleaned.sort(key=lambda x: x["range"], reverse=True)

        distances = [t["range"] for t in cleaned]
        types_list = [t["type"] for t in cleaned]
        num_targets = len(cleaned)

    else:
        # Original behavior
        distances = np.linspace(dist_max, dist_min, num_targets).tolist()
        types_list = [pick_target(target_type) for _ in range(num_targets)]

    # Page setup
    page_w, page_h = landscape(letter)
    margin_pts = page_margin * 72

    c = canvas.Canvas(file_path, pagesize=(page_w, page_h))

    # Even spacing
    x_positions = np.linspace(margin_pts, page_w - margin_pts, num_targets)
    y_positions = np.linspace(margin_pts, page_h - margin_pts, num_targets)

    for idx in range(num_targets):
        sim_dist = distances[idx]
        this_type = types_list[idx]

        draw, w_pt, h_pt = build_target_drawing(this_type, wall_distance, sim_dist)

        x = x_positions[idx]
        y = y_positions[idx]

        # Clamp X
        if x + w_pt > page_w - margin_pts:
            x = page_w - margin_pts - w_pt
        if x < margin_pts:
            x = margin_pts

        # Clamp Y
        if y + h_pt > page_h - margin_pts:
            y = page_h - margin_pts - h_pt
        if y < margin_pts:
            y = margin_pts

        renderPDF.draw(draw, c, x, y)

    c.save()
    
    
    # ---------------------------------------------------------
# App Config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Dry-Fire Stage Generator",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("Dry-Fire Stage Generator")

# ---------------------------------------------------------
# Sidebar Controls (maps cleanly to Qt widgets)
# ---------------------------------------------------------
with st.sidebar:
    st.header("Stage Parameters")

    wall_distance = st.number_input(
        "Actual Distance (yards)",
        min_value=0,
        max_value=200,
        value=10,
        step=1
    )

    num_targets = st.slider(
        "Number of Targets",
        min_value=1,
        max_value=8,
        value=4
    )

    num_positions = st.slider(
        "Number of Positions",
        min_value=1,
        max_value=6,
        value=2
    )

    difficulty = st.selectbox(
        "Difficulty",
        ["Easy", "Medium", "Hard"]
    )

    generate_clicked = st.button("Generate Stage")

# ---------------------------------------------------------
# Session State (replaces Qt object persistence)
# ---------------------------------------------------------
if "stage_data" not in st.session_state:
    st.session_state.stage_data = None
if "targets" not in st.session_state:
    st.session_state.targets = None

# ---------------------------------------------------------
# Stage Generation
# ---------------------------------------------------------
if generate_clicked:
    difficulty_map = {"Easy": 1, "Medium": 2, "Hard": 3}

    data, targets = generate_stage(num_targets=num_targets)

    st.session_state.stage_data = data
    st.session_state.targets = targets

# ---------------------------------------------------------
# Output Display
# ---------------------------------------------------------
if st.session_state.stage_data:
    st.subheader("Stage Generated")

    st.markdown(st.session_state.stage_data["brief"])

# ---------------------------------------------------------
# PDF Export
# ---------------------------------------------------------
if st.session_state.targets:
    st.divider()
    st.subheader("Export")

    if st.button("Export Printable Targets (PDF)"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            export_pdf(
                tmp.name,
                targets=st.session_state.targets,
                wall_distance=wall_distance
            )
            tmp_path = tmp.name

        with open(tmp_path, "rb") as f:
            st.download_button(
                label="Download PDF",
                data=f,
                file_name="targets.pdf",
                mime="application/pdf"
            )

        os.unlink(tmp_path)