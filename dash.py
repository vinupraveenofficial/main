import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import numpy as np
from datetime import datetime, timedelta

# === CONFIGURATION ===
LOG_FILE = "detections_log.csv"
INPUT_FOLDER = "detected_images"
DEFAULT_LAT, DEFAULT_LON = 30.767, 76.575
AUTO_REFRESH_INTERVAL = 30  # seconds

# --- Page setup ---
st.set_page_config(page_title="Emission Dashboard", layout="wide", page_icon="🌍")
st.title("🌍 Emission Monitoring & Forecast Dashboard")

# Auto-refresh
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = datetime.now()
elif (datetime.now() - st.session_state.last_refresh).seconds > AUTO_REFRESH_INTERVAL:
    st.session_state.last_refresh = datetime.now()
    st.experimental_rerun()

# --- Load data ---
if not os.path.exists(LOG_FILE):
    st.warning("No detection log found. Waiting for data...")
    st.stop()

df = pd.read_csv(LOG_FILE)
df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
df = df[df["DateTime"] >= datetime.now() - timedelta(hours=24)]
df["Latitude"] = DEFAULT_LAT
df["Longitude"] = DEFAULT_LON

# --- Layout ---
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

# --- Panel 1: Recent Detections ---
with col1:
    st.subheader("📷 Recent Detections")
    abs_input_folder = os.path.abspath(INPUT_FOLDER)
    image_files = sorted(
        [os.path.join(abs_input_folder, f) for f in os.listdir(abs_input_folder)
         if f.lower().endswith((".jpg", ".png", ".jpeg"))],
        key=os.path.getmtime,
        reverse=True
    )
    if image_files:
        n_images = min(len(image_files), 6)
        cols = st.columns(3)
        for i, img_path in enumerate(image_files[:n_images]):
            with cols[i % 3]:
                st.image(img_path, caption=os.path.basename(img_path), use_container_width=True)
    else:
        st.info("No detected images yet.")

# --- Panel 2: Detection Trends + Wind Speed ---
with col2:
    st.subheader("📈 Detection Trends")
    if not df.empty:
        df_count = df.groupby(df["DateTime"].dt.hour).size().reset_index(name="Detections")
        fig_count = px.bar(df_count, x="DateTime", y="Detections", text="Detections")
        fig_count.update_traces(textposition="outside")
        st.plotly_chart(fig_count, use_container_width=True)

    st.subheader("🌬️ Wind Speed (last hour)")
    df_wind = df.dropna(subset=["WindSpeed_kmh"])
    if not df_wind.empty:
        fig_wind = px.line(df_wind, x="DateTime", y="WindSpeed_kmh", markers=True,
                           title="Wind Speed (km/h) at Detection Times")
        st.plotly_chart(fig_wind, use_container_width=True)

# --- Panel 3: Geo Hotspot Map with Wind Arrows ---
with col3:
    st.subheader("🌍 Geo Hotspot Map with Wind Direction")

    if not df.empty:
        df_map = df.groupby(["Latitude", "Longitude"]).agg(
            Emission_Count=("Filename", "count"),
            WindDir_deg=("WindDir_deg", "mean")  # average wind direction at hotspot
        ).reset_index()

        # Base scatter points for hotspots
        fig_map = go.Figure()
        fig_map.add_trace(go.Scattermapbox(
            lat=df_map["Latitude"],
            lon=df_map["Longitude"],
            mode="markers",
            marker=go.scattermapbox.Marker(
                size=df_map["Emission_Count"]*10,
                color=df_map["Emission_Count"],
                colorscale="Reds",
                cmin=1,
                cmax=df_map["Emission_Count"].max()
            ),
            text=df_map["Emission_Count"],
            hoverinfo="text+lat+lon",
            name="Hotspots"
        ))

        # Add wind arrows (line + triangle marker)
        scale = 0.001  # length of arrow line
        arrow_size = 14  # marker size for arrowhead
        for _, row in df_map.iterrows():
            lat0, lon0 = row["Latitude"], row["Longitude"]
            deg = row["WindDir_deg"]
            rad = np.deg2rad(deg)

            # Line coordinates
            lat1 = lat0 + scale * np.cos(rad)
            lon1 = lon0 + scale * np.sin(rad)

            # Line
            fig_map.add_trace(go.Scattermapbox(
                lat=[lat0, lat1],
                lon=[lon0, lon1],
                mode="lines",
                line=dict(color="blue", width=2),
                showlegend=False
            ))

            # Arrowhead
            fig_map.add_trace(go.Scattermapbox(
                lat=[lat1],
                lon=[lon1],
                mode="markers",
                marker=dict(
                    size=arrow_size,
                    color="blue",
                    symbol="triangle-up",
                    angle=deg  # rotate to match wind direction
                ),
                showlegend=False
            ))

        fig_map.update_layout(
            mapbox_style="open-street-map",
            mapbox_zoom=15,
            mapbox_center={"lat": DEFAULT_LAT, "lon": DEFAULT_LON},
            margin={"l":0, "r":0, "t":0, "b":0}
        )

        st.plotly_chart(fig_map, use_container_width=True)

# --- Panel 4: Correlation & Analytics ---
with col4:
    st.subheader("🧠 Emission Analytics")
    st.markdown("""
        - Number of boxes per detection = emission severity  
        - Average confidence of YOLO detections  
        - Wind-emission correlation plots
    """)

    # Emission severity vs Wind Speed
    df_corr = df.dropna(subset=["Num_Boxes", "WindSpeed_kmh"])
    if not df_corr.empty:
        fig_corr = px.scatter(df_corr, x="WindSpeed_kmh", y="Num_Boxes",
                              title="Emission Severity vs Wind Speed")
        st.plotly_chart(fig_corr, use_container_width=True)

    # Emission count vs Wind Direction
    df_dir = df.groupby("WindDir_compass").size().reset_index(name="Count")
    if not df_dir.empty:
        fig_dir = px.bar(df_dir, x="WindDir_compass", y="Count", title="Emission Count by Wind Direction")
        st.plotly_chart(fig_dir, use_container_width=True)

st.divider()
st.markdown("© 2025 Emission Monitoring System | Powered by AI + IoT + Open Meteo")
