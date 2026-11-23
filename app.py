# app.py:  Week 11-12: API Client + Model Server
# Part A: Jolpica F1 API client (fetches live race results)
# Part B: F1 Podium Prediction client (talks to our FastAPI server)
# Soumay Patidar | DATS 6401

import streamlit as st
import requests
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Week 11-12: F1 APIs & Model Serving", layout="wide")

# helper: safe JSON fetch with the three checks from class
@st.cache_data(ttl=300)
def get_json(url, params=None, timeout=20):
    """fetch JSON with the three checks:
    1. status code  2. content type  3. shape"""
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()

    ctype = r.headers.get("content-type", "")
    if "json" not in ctype:
        raise ValueError(f"expected JSON, got {ctype!r}: {r.text[:120]}")

    payload = r.json()
    if not isinstance(payload, dict):
        raise ValueError(f"expected dict, got {type(payload).__name__}")
    return payload


tab_a, tab_b = st.tabs(["Part A — F1 Season Results (API Client)",
                         "Part B — Podium Predictor (Model Server)"])


# PART A: Jolpica F1 API Client

with tab_a:
    st.title("F1 Season Explorer")
    st.write(
        "This pulls live data from the Jolpica F1 API, which is basically the "
        "successor to the Ergast API. Every time you change the season or the "
        "number of results it fires a new request — nothing is pre-downloaded. "
        "Results are cached for 5 minutes so repeating a combo doesn't hit the "
        "server again."
    )

    API_BASE = "https://api.jolpi.ca/ergast/f1"

    col1, col2 = st.columns(2)
    with col1:
        season = st.slider("Season", 1950, 2024, 2024)
    with col2:
        limit = st.selectbox("Max results per request", [30, 50, 100], index=0)

    # show what URL we're about to hit
    results_url = f"{API_BASE}/{season}/results.json"
    st.caption(f"Request: `{results_url}?limit={limit}`")

    try:
        data = get_json(results_url, params={"limit": limit})
    except requests.exceptions.Timeout:
        st.error("The API took too long to respond — try again in a sec.")
        st.stop()
    except requests.exceptions.HTTPError as e:
        st.error(f"API returned an error: {e}")
        st.stop()
    except requests.exceptions.ConnectionError:
        st.error(
            "Can't reach the Jolpica F1 API right now. "
            "Check your internet or try again later."
        )
        st.stop()
    except ValueError as e:
        st.error(f"Got a response but it wasn't valid JSON: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Something went wrong: {e}")
        st.stop()

    # dig into the nested JSON — Ergast-style APIs bury the data pretty deep
    try:
        race_table = data["MRData"]["RaceTable"]
        races = race_table.get("Races", [])
    except (KeyError, TypeError):
        st.error("The JSON structure wasn't what we expected from the Ergast API.")
        st.stop()

    if not races:
        st.warning(f"No results found for the {season} season. "
                    "Maybe it hasn't happened yet or the API doesn't have it.")
        st.stop()

    # flatten the nested JSON into rows
    rows = []
    for race in races:
        race_name = race.get("raceName", "Unknown")
        race_round = race.get("round", "?")
        for result in race.get("Results", []):
            driver = result.get("Driver", {})
            constructor = result.get("Constructor", {})
            rows.append({
                "round": int(race_round),
                "race": race_name,
                "position": int(result.get("position", 0)),
                "driver": f"{driver.get('givenName', '')} {driver.get('familyName', '')}",
                "driver_code": driver.get("code", "???"),
                "constructor": constructor.get("name", "Unknown"),
                "grid": int(result.get("grid", 0)),
                "points": float(result.get("points", 0)),
                "laps": int(result.get("laps", 0)),
                "status": result.get("status", "Unknown"),
            })

    df = pd.DataFrame(rows)
    st.metric("Race entries loaded", len(df))

    # points per driver
    st.subheader("Points by Driver")
    driver_points = (df.groupby("driver_code")["points"]
                     .sum()
                     .sort_values(ascending=False)
                     .head(15)
                     .reset_index())
    driver_points.columns = ["Driver", "Points"]

    fig_bar = px.bar(
        driver_points, x="Driver", y="Points",
        color="Points", color_continuous_scale="YlOrRd",
        title=f"{season} Season — Top Drivers by Points (first {limit} results)",
    )
    fig_bar.update_layout(height=400)
    st.plotly_chart(fig_bar, use_container_width=True)

    # grid vs finish position
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Grid → Finish Position")
        fig_scatter = px.scatter(
            df, x="grid", y="position",
            color="constructor",
            hover_data=["driver", "race"],
            title="Did they gain or lose places?",
        )
        fig_scatter.update_layout(
            height=400,
            xaxis_title="Grid Position",
            yaxis_title="Finish Position",
        )
        # add a diagonal line — above it means they lost places
        max_pos = max(df["grid"].max(), df["position"].max())
        fig_scatter.add_shape(
            type="line", x0=1, y0=1, x1=max_pos, y1=max_pos,
            line=dict(color="gray", dash="dash"),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_b:
        st.subheader("Points by Constructor")
        constructor_pts = (df.groupby("constructor")["points"]
                          .sum()
                          .sort_values(ascending=False)
                          .reset_index())
        constructor_pts.columns = ["Constructor", "Points"]

        fig_team = px.bar(
            constructor_pts, x="Points", y="Constructor",
            orientation="h",
            color="Points", color_continuous_scale="Viridis",
            title=f"{season} Constructor Points",
        )
        fig_team.update_layout(height=400, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_team, use_container_width=True)

    # ── raw data ──
    with st.expander("Raw data from the API"):
        st.dataframe(df, use_container_width=True)

    st.write("---")
    st.caption(
        "Data: [Jolpica F1 API](https://api.jolpi.ca/ergast/f1/) "
        "(Ergast successor). Cached for 5 min (`@st.cache_data(ttl=300)`). "
        "All requests have `timeout=20`."
    )

# PART B: F1 Podium Prediction Client

with tab_b:
    st.title("F1 Podium Predictor")
    st.write(
        "This tab talks to the FastAPI server running locally. "
        "It sends grid position, constructor, circuit and race info "
        "to `/predict` and shows the probability of a podium finish. "
        "Start the server first: `uvicorn server:app --reload`"
    )

    server_url = st.text_input(
        "Server URL", value="http://127.0.0.1:8000",
        help="Where your FastAPI server is running",
    )

    # try to fetch the list of constructors from the server
    known_constructors = None
    try:
        r = requests.get(f"{server_url}/constructors", timeout=5)
        if r.status_code == 200:
            known_constructors = r.json().get("constructors", [])
    except Exception:
        pass

    st.subheader("Race Entry Details")
    c1, c2 = st.columns(2)
    with c1:
        grid = st.slider("Grid position", 1, 20, 3)
        if known_constructors:
            # show the ones most people care about first
            popular = ["red_bull", "mercedes", "ferrari", "mclaren",
                       "aston_martin", "alpine", "williams", "haas",
                       "alphatauri", "alfa"]
            sorted_constructors = (
                [c for c in popular if c in known_constructors]
                + [c for c in known_constructors if c not in popular]
            )
            constructor = st.selectbox("Constructor", sorted_constructors)
        else:
            constructor = st.text_input("Constructor", value="red_bull",
                                        help="e.g. red_bull, mercedes, ferrari")

    with c2:
        circuit_id = st.number_input("Circuit ID", 1, 80, 9,
                                     help="Ergast circuit ID (9 = Silverstone, 6 = Monaco, 1 = Albert Park)")
        race_round = st.slider("Race round", 1, 24, 10)
        laps = st.number_input("Number of laps", 10, 80, 57)

    if st.button("Predict Podium", type="primary"):
        payload = {
            "grid": grid,
            "constructor": constructor,
            "circuit_id": circuit_id,
            "race_round": race_round,
            "laps": laps,
        }

        try:
            r = requests.post(
                f"{server_url}/predict",
                json=payload,
                timeout=10,
            )

            if r.status_code == 200:
                result = r.json()

                podium_pct = result["podium_probability"]

                if result["prediction"] == "Podium":
                    st.success(f"Podium! The model gives this a **{podium_pct:.0%}** chance.")
                else:
                    st.warning(f"Probably not a podium — only **{podium_pct:.0%}** chance.")

                # metrics
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Podium Probability", f"{podium_pct:.1%}")
                with m2:
                    st.metric("Grid Position", f"P{result['grid']}")
                with m3:
                    st.metric("Constructor", result["constructor"])

                # probability bar chart
                prob_df = pd.DataFrame({
                    "Outcome": ["Podium", "No Podium"],
                    "Probability": [result["podium_probability"],
                                    result["no_podium_probability"]],
                })
                fig_prob = px.bar(
                    prob_df, x="Outcome", y="Probability",
                    color="Outcome",
                    color_discrete_map={"Podium": "#2ecc71", "No Podium": "#e74c3c"},
                    title="Prediction Probabilities",
                )
                fig_prob.update_layout(
                    height=350,
                    yaxis=dict(range=[0, 1], tickformat=".0%"),
                    showlegend=False,
                )
                st.plotly_chart(fig_prob, use_container_width=True)

                with st.expander("Raw response"):
                    st.json(result)

            elif r.status_code == 400:
                st.error(f"Bad request: {r.json().get('detail', 'Unknown error')}")
            elif r.status_code == 422:
                st.error(f"Invalid input: {r.json()}")
            else:
                st.error(f"Server returned status {r.status_code}")

        except requests.exceptions.ConnectionError:
            st.error(
                "Can't connect to the server. Make sure it's running:\n\n"
                "`uvicorn server:app --reload`"
            )
        except requests.exceptions.Timeout:
            st.error("Server took too long to respond.")
        except Exception as e:
            st.error(f"Something went wrong: {e}")

    # compare a few scenarios
    st.write("---")
    st.subheader("Quick Compare")
    st.write(
        "Hit the button below to send a few different scenarios at once — "
        "shows how much grid position and constructor matter."
    )

    if st.button("Run comparison"):
        scenarios = [
            {"grid": 1, "constructor": "red_bull", "circuit_id": 9, "race_round": 10, "laps": 52,
             "label": "Red Bull P1"},
            {"grid": 1, "constructor": "mercedes", "circuit_id": 9, "race_round": 10, "laps": 52,
             "label": "Mercedes P1"},
            {"grid": 1, "constructor": "ferrari", "circuit_id": 9, "race_round": 10, "laps": 52,
             "label": "Ferrari P1"},
            {"grid": 10, "constructor": "red_bull", "circuit_id": 9, "race_round": 10, "laps": 52,
             "label": "Red Bull P10"},
            {"grid": 1, "constructor": "williams", "circuit_id": 9, "race_round": 10, "laps": 52,
             "label": "Williams P1"},
            {"grid": 15, "constructor": "williams", "circuit_id": 9, "race_round": 10, "laps": 52,
             "label": "Williams P15"},
        ]

        compare_rows = []
        for s in scenarios:
            label = s.pop("label")
            try:
                r = requests.post(f"{server_url}/predict", json=s, timeout=5)
                if r.status_code == 200:
                    result = r.json()
                    compare_rows.append({
                        "Scenario": label,
                        "Podium %": result["podium_probability"],
                    })
            except Exception:
                compare_rows.append({"Scenario": label, "Podium %": None})

        if compare_rows:
            comp_df = pd.DataFrame(compare_rows)
            fig_comp = px.bar(
                comp_df, x="Scenario", y="Podium %",
                color="Podium %", color_continuous_scale="RdYlGn",
                title="Podium probability across scenarios (all at Silverstone)",
            )
            fig_comp.update_layout(
                height=400,
                yaxis=dict(range=[0, 1], tickformat=".0%"),
            )
            st.plotly_chart(fig_comp, use_container_width=True)

    # health check
    st.write("---")
    if st.button("Check server health"):
        try:
            r = requests.get(f"{server_url}/health", timeout=5)
            r.raise_for_status()
            st.success(f"Server is up! {r.json()}")
        except Exception as e:
            st.error(f"Server not reachable: {e}")
