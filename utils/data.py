import pandas as pd
import requests
import streamlit as st

NYC_DOHMH_API = "https://data.cityofnewyork.us/resource/43nn-pn8j.json"


@st.cache_data(ttl=86400, show_spinner=False)
def load_nyc_base(limit=8000):
    params = {
        "$limit": limit,
        "$where": "grade IN('A','B','C') AND cuisine_description IS NOT NULL AND dba IS NOT NULL",
        "$select": "camis,dba,boro,building,street,zipcode,cuisine_description,grade,score,latitude,longitude",
        "$order": "grade ASC",
    }

    response = requests.get(NYC_DOHMH_API, params=params, timeout=30)
    response.raise_for_status()
    df = pd.DataFrame(response.json())

    if df.empty:
        return df

    df = df.drop_duplicates(subset=["camis"], keep="first").copy()
    df["dba"] = df["dba"].fillna("").str.title().str.strip()
    df["cuisine"] = df["cuisine_description"].fillna("").str.strip()
    df["boro"] = df["boro"].fillna("").str.title().str.strip()
    df["address"] = (
        df.get("building", pd.Series([""] * len(df))).fillna("").astype(str)
        + " "
        + df.get("street", pd.Series([""] * len(df))).fillna("").astype(str)
        + ", "
        + df.get("boro", pd.Series([""] * len(df))).fillna("").astype(str)
        + ", NY "
        + df.get("zipcode", pd.Series([""] * len(df))).fillna("").astype(str)
    ).str.strip(", ")
    df["grade"] = df.get("grade", pd.Series(["N/A"] * len(df))).fillna("N/A")
    df["score"] = pd.to_numeric(df.get("score", 0), errors="coerce").fillna(0).astype(int)
    df["lat"] = pd.to_numeric(df.get("latitude", None), errors="coerce")
    df["lon"] = pd.to_numeric(df.get("longitude", None), errors="coerce")
    return df.reset_index(drop=True)


def load_nyc_base_safe(limit=8000):
    try:
        return load_nyc_base(limit=limit)
    except Exception:
        return pd.DataFrame()
