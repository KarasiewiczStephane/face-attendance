"""Streamlit dashboard for face attendance system visualization.

Displays attendance log tables, daily attendance charts, weekly summary
reports, and face database gallery using synthetic demo data.

Run with: streamlit run src/dashboard/app.py
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

EMPLOYEE_NAMES = [
    "Alice Johnson",
    "Bob Smith",
    "Carol Williams",
    "David Brown",
    "Eva Martinez",
    "Frank Lee",
    "Grace Kim",
    "Henry Chen",
    "Iris Patel",
    "Jack Wilson",
    "Karen Davis",
    "Leo Garcia",
]


def generate_attendance_log(
    num_days: int = 14,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic attendance log data.

    Args:
        num_days: Number of days of attendance data to generate.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns: id, person_id, name, timestamp, date,
        check_in_time, confidence, liveness_score, status.
    """
    rng = np.random.default_rng(seed)
    records = []
    record_id = 1
    base_date = datetime(2024, 11, 18)

    for day_offset in range(num_days):
        current_date = base_date + timedelta(days=day_offset)
        if current_date.weekday() >= 5:
            continue

        for person_idx, name in enumerate(EMPLOYEE_NAMES):
            absent_prob = 0.08
            if rng.random() < absent_prob:
                continue

            base_hour = 8
            base_minute = 30 + int(rng.normal(15, 12))
            base_minute = max(0, min(59, base_minute))
            if base_minute > 45:
                base_hour = 9
                base_minute = base_minute - 60 + rng.integers(0, 20)
                base_minute = max(0, min(59, base_minute))

            check_in = current_date.replace(
                hour=base_hour, minute=base_minute, second=rng.integers(0, 60)
            )

            is_late = check_in.hour > 9 or (check_in.hour == 9 and check_in.minute > 15)

            confidence = round(0.85 + rng.random() * 0.14, 3)
            liveness = round(0.90 + rng.random() * 0.09, 3)

            records.append(
                {
                    "id": record_id,
                    "person_id": person_idx + 1,
                    "name": name,
                    "timestamp": check_in.strftime("%Y-%m-%d %H:%M:%S"),
                    "date": current_date.strftime("%Y-%m-%d"),
                    "check_in_time": check_in.strftime("%H:%M:%S"),
                    "confidence": confidence,
                    "liveness_score": liveness,
                    "status": "late" if is_late else "present",
                }
            )
            record_id += 1

    return pd.DataFrame(records)


def generate_face_gallery(seed: int = 42) -> pd.DataFrame:
    """Generate synthetic face database entries.

    Args:
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with person_id, name, enrolled_date, embedding_count,
        last_seen, and is_active columns.
    """
    rng = np.random.default_rng(seed)
    records = []

    for i, name in enumerate(EMPLOYEE_NAMES):
        enrolled = datetime(2024, 1, 1) + timedelta(days=int(rng.integers(0, 200)))
        last_seen = datetime(2024, 11, 28) - timedelta(days=int(rng.integers(0, 5)))

        records.append(
            {
                "person_id": i + 1,
                "name": name,
                "enrolled_date": enrolled.strftime("%Y-%m-%d"),
                "embedding_count": rng.integers(3, 12),
                "last_seen": last_seen.strftime("%Y-%m-%d"),
                "is_active": True,
            }
        )

    return pd.DataFrame(records)


def render_attendance_log(df: pd.DataFrame) -> None:
    """Render the attendance log table section."""
    st.header("Attendance Log")

    col1, col2, col3 = st.columns(3)
    dates = sorted(df["date"].unique())

    with col1:
        selected_date = st.selectbox("Select Date", dates, index=len(dates) - 1)

    filtered = df[df["date"] == selected_date]

    with col2:
        status_filter = st.multiselect(
            "Filter by Status",
            ["present", "late"],
            default=["present", "late"],
        )

    filtered = filtered[filtered["status"].isin(status_filter)]

    with col3:
        st.metric("Records", len(filtered))

    present_count = len(filtered[filtered["status"] == "present"])
    late_count = len(filtered[filtered["status"] == "late"])
    absent_count = len(EMPLOYEE_NAMES) - present_count - late_count

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Employees", len(EMPLOYEE_NAMES))
    m2.metric("Present", present_count)
    m3.metric("Late", late_count)
    m4.metric("Absent", absent_count)

    def highlight_status(row: pd.Series) -> list[str]:
        if row["status"] == "late":
            return ["background-color: #fff3cd"] * len(row)
        return [""] * len(row)

    display_cols = [
        "name",
        "check_in_time",
        "status",
        "confidence",
        "liveness_score",
    ]
    styled = (
        filtered[display_cols]
        .style.apply(highlight_status, axis=1)
        .format({"confidence": "{:.1%}", "liveness_score": "{:.1%}"})
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)


def render_daily_chart(df: pd.DataFrame) -> None:
    """Render daily attendance chart section."""
    st.header("Daily Attendance Chart")

    daily = df.groupby(["date", "status"]).size().reset_index(name="count")

    dates = sorted(df["date"].unique())
    absent_records = []
    for date in dates:
        day_df = df[df["date"] == date]
        total_present = len(day_df)
        absent_count = len(EMPLOYEE_NAMES) - total_present
        if absent_count > 0:
            absent_records.append({"date": date, "status": "absent", "count": absent_count})

    if absent_records:
        absent_df = pd.DataFrame(absent_records)
        daily = pd.concat([daily, absent_df], ignore_index=True)

    color_map = {"present": "#4CAF50", "late": "#FFC107", "absent": "#F44336"}

    fig = px.bar(
        daily,
        x="date",
        y="count",
        color="status",
        color_discrete_map=color_map,
        title="Daily Attendance Breakdown",
        labels={"count": "Number of People", "date": "Date"},
        barmode="stack",
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    avg_check_in = df.groupby("date")["check_in_time"].apply(
        lambda x: pd.to_datetime(x).mean().strftime("%H:%M")
    )
    avg_df = avg_check_in.reset_index()
    avg_df.columns = ["date", "avg_check_in"]

    fig2 = px.line(
        avg_df,
        x="date",
        y="avg_check_in",
        title="Average Check-in Time by Day",
        markers=True,
        labels={"avg_check_in": "Average Check-in", "date": "Date"},
    )
    fig2.update_layout(height=350)
    st.plotly_chart(fig2, use_container_width=True)


def render_weekly_summary(df: pd.DataFrame) -> None:
    """Render weekly summary report section."""
    st.header("Weekly Summary Report")

    df_copy = df.copy()
    df_copy["week"] = pd.to_datetime(df_copy["date"]).dt.isocalendar().week

    weeks = sorted(df_copy["week"].unique())
    selected_week = st.selectbox("Select Week", weeks, index=len(weeks) - 1)

    week_df = df_copy[df_copy["week"] == selected_week]
    dates_in_week = sorted(week_df["date"].unique())

    summary_records = []
    for name in EMPLOYEE_NAMES:
        person_week = week_df[week_df["name"] == name]
        days_present = len(person_week[person_week["status"] == "present"])
        days_late = len(person_week[person_week["status"] == "late"])
        days_absent = len(dates_in_week) - days_present - days_late
        attendance_rate = (days_present + days_late) / max(len(dates_in_week), 1) * 100

        avg_confidence = person_week["confidence"].mean() if len(person_week) > 0 else 0

        summary_records.append(
            {
                "Name": name,
                "Days Present": days_present,
                "Days Late": days_late,
                "Days Absent": days_absent,
                "Attendance Rate": attendance_rate,
                "Avg Confidence": avg_confidence,
            }
        )

    summary_df = pd.DataFrame(summary_records)

    col1, col2, col3 = st.columns(3)
    overall_rate = summary_df["Attendance Rate"].mean()
    col1.metric("Overall Attendance Rate", f"{overall_rate:.1f}%")
    col2.metric("Working Days", len(dates_in_week))
    col3.metric(
        "Perfect Attendance",
        len(summary_df[summary_df["Attendance Rate"] == 100]),
    )

    styled = summary_df.style.format({"Attendance Rate": "{:.1f}%", "Avg Confidence": "{:.1%}"})
    st.dataframe(styled, use_container_width=True, hide_index=True)

    fig = px.bar(
        summary_df.sort_values("Attendance Rate"),
        x="Attendance Rate",
        y="Name",
        orientation="h",
        title="Weekly Attendance Rate by Employee",
        color="Attendance Rate",
        color_continuous_scale="RdYlGn",
        labels={"Attendance Rate": "Attendance (%)", "Name": "Employee"},
    )
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

    status_counts = week_df["status"].value_counts()
    fig2 = go.Figure(
        data=[
            go.Pie(
                labels=status_counts.index.tolist(),
                values=status_counts.values.tolist(),
                marker={"colors": ["#4CAF50", "#FFC107"]},
                hole=0.4,
            )
        ]
    )
    fig2.update_layout(title="Weekly Status Distribution", height=350)
    st.plotly_chart(fig2, use_container_width=True)


def render_face_gallery(df: pd.DataFrame) -> None:
    """Render face database gallery section."""
    st.header("Face Database Gallery")

    col1, col2, col3 = st.columns(3)
    col1.metric("Registered Faces", len(df))
    col2.metric("Active Members", len(df[df["is_active"]]))
    col3.metric("Total Embeddings", df["embedding_count"].sum())

    cols = st.columns(4)
    for idx, (_, row) in enumerate(df.iterrows()):
        with cols[idx % 4]:
            initials = "".join(word[0] for word in row["name"].split() if word)

            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, #667eea, #764ba2);
                    border-radius: 12px;
                    padding: 20px;
                    text-align: center;
                    margin-bottom: 10px;
                    color: white;
                ">
                    <div style="
                        font-size: 36px;
                        font-weight: bold;
                        margin-bottom: 8px;
                    ">{initials}</div>
                    <div style="font-size: 14px; font-weight: 600;">
                        {row["name"]}
                    </div>
                    <div style="font-size: 11px; opacity: 0.8;">
                        ID: {row["person_id"]} |
                        Embeddings: {row["embedding_count"]}
                    </div>
                    <div style="font-size: 11px; opacity: 0.8;">
                        Last seen: {row["last_seen"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


@st.cache_data
def load_demo_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load all synthetic demo data.

    Returns:
        Tuple of (attendance_log, face_gallery).
    """
    return generate_attendance_log(), generate_face_gallery()


def main() -> None:
    """Run the Streamlit dashboard application."""
    st.set_page_config(
        page_title="Face Attendance Dashboard",
        page_icon="👤",
        layout="wide",
    )

    st.title("Face Attendance Dashboard")
    st.markdown(
        "Attendance tracking with face recognition: log tables, daily charts, "
        "weekly summaries, and registered face database."
    )

    attendance_df, gallery_df = load_demo_data()

    render_attendance_log(attendance_df)
    st.divider()
    render_daily_chart(attendance_df)
    st.divider()
    render_weekly_summary(attendance_df)
    st.divider()
    render_face_gallery(gallery_df)


if __name__ == "__main__":
    main()
