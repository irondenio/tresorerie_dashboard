import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


st.image("image/copyrights.png", caption="© 2026 ", width=100)

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Dashboard Trésorerie", layout="wide")
st.title("Tableau de bord de trésorerie avec prévision ML")

st.subheader("Anthony DJOUMBISSI")



# -----------------------------
# HELPERS
# -----------------------------
def normalize_type(x: str) -> str:
    if pd.isna(x):
        return ""
    x = str(x).strip().lower()
    if "enc" in x:
        return "Encaissement"
    if "dec" in x or "déc" in x:
        return "Decaissement"
    return x

def prepare_transactions(df: pd.DataFrame) -> pd.DataFrame:
    # Normaliser noms de colonnes
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"date", "type", "categorie", "montant"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans le CSV: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["type"] = df["type"].apply(normalize_type)
    df["categorie"] = df["categorie"].astype(str).str.strip()
    df["montant"] = pd.to_numeric(df["montant"], errors="coerce")

    df = df.dropna(subset=["date", "type", "categorie", "montant"])
    df = df[df["montant"] >= 0]

    # Montant signé: encaissements positifs, décaissements négatifs
    df["montant_signe"] = np.where(df["type"] == "Encaissement", df["montant"], -df["montant"])
    return df

def daily_cashflow(df: pd.DataFrame) -> pd.DataFrame:
    daily = (
        df.groupby("date", as_index=False)["montant_signe"]
          .sum()
          .rename(columns={"date": "ds", "montant_signe": "y"})
          .sort_values("ds")
    )
    return daily

def fmt_money(x: float) -> str:
    try:
        return f"{x:,.0f}".replace(",", " ")
    except Exception:
        return str(x)

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features calendaires et tendance pour la prévision:
    - t: index temporel
    - dow: jour de semaine (0..6) + sin/cos
    - month: mois (1..12) + sin/cos
    """
    out = df.copy()
    out["ds"] = pd.to_datetime(out["ds"])
    out = out.sort_values("ds")
    out["t"] = (out["ds"] - out["ds"].min()).dt.days.astype(float)

    out["dow"] = out["ds"].dt.dayofweek.astype(int)
    out["month"] = out["ds"].dt.month.astype(int)

    # Saison hebdo
    out["dow_sin"] = np.sin(2 * np.pi * out["dow"] / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * out["dow"] / 7.0)

    # Saison annuelle approximée par mois
    out["mon_sin"] = np.sin(2 * np.pi * (out["month"] - 1) / 12.0)
    out["mon_cos"] = np.cos(2 * np.pi * (out["month"] - 1) / 12.0)

    return out

def fit_forecast_ml(daily: pd.DataFrame, horizon_days: int, interval_width: float) -> pd.DataFrame:
    """
    Prévision cloud-friendly:
    - Ridge regression sur features calendaires + tendance
    - Intervalle basé sur écart-type des résidus (approx)
    interval_width: ex 0.80 -> z ~ 1.2816 (approx normal)
    """
    df = add_time_features(daily)

    feature_cols = ["t", "dow_sin", "dow_cos", "mon_sin", "mon_cos"]
    X = df[feature_cols].values
    y = df["y"].values.astype(float)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0, random_state=42))
    ])
    model.fit(X, y)

    # Résidus sur historique (pour incertitude)
    y_pred_hist = model.predict(X)
    resid = y - y_pred_hist
    sigma = float(np.std(resid)) if len(resid) > 5 else float(np.std(y)) if len(y) > 1 else 0.0

    # Z-score approximatif pour intervalle central (normal)
    # 0.80 -> 1.2816 ; 0.90 -> 1.6449 ; 0.95 -> 1.96
    z_map = {0.50: 0.674, 0.55: 0.755, 0.60: 0.842, 0.65: 0.935, 0.70: 1.036,
             0.75: 1.150, 0.80: 1.282, 0.85: 1.440, 0.90: 1.645, 0.95: 1.960}
    # Arrondir à 0.05 près pour mapper
    iw = round(interval_width / 0.05) * 0.05
    z = z_map.get(iw, 1.282)

    # Construire futur
    last_date = df["ds"].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon_days, freq="D")

    future = pd.DataFrame({"ds": future_dates})
    # on concat pour générer t cohérent
    all_df = pd.concat([df[["ds"]], future], ignore_index=True)
    all_feat = add_time_features(all_df)

    X_all = all_feat[feature_cols].values
    yhat_all = model.predict(X_all)

    # Séparer historique + futur dans le format Prophet-like
    out = pd.DataFrame({
        "ds": all_feat["ds"],
        "yhat": yhat_all
    })

    out["yhat_lower"] = out["yhat"] - z * sigma
    out["yhat_upper"] = out["yhat"] + z * sigma

    return out

def build_scenarios(fcst: pd.DataFrame, opt_pct: float, pess_pct: float) -> pd.DataFrame:
    base = fcst.copy()
    base["scenario"] = "Base"

    optimistic = fcst.copy()
    optimistic[["yhat", "yhat_lower", "yhat_upper"]] *= (1.0 + opt_pct)
    optimistic["scenario"] = "Optimiste"

    pessimistic = fcst.copy()
    pessimistic[["yhat", "yhat_lower", "yhat_upper"]] *= (1.0 - pess_pct)  # pess_pct est une baisse positive
    pessimistic["scenario"] = "Pessimiste"

    return pd.concat([base, optimistic, pessimistic], ignore_index=True)

def treasury_curve_from_cashflow(cashflow_df: pd.DataFrame, initial_balance: float) -> pd.DataFrame:
    cashflow_df = cashflow_df.sort_values(["scenario", "ds"]).copy()
    cashflow_df["tresorerie"] = cashflow_df.groupby("scenario")["yhat"].cumsum() + initial_balance
    cashflow_df["tresorerie_lower"] = cashflow_df.groupby("scenario")["yhat_lower"].cumsum() + initial_balance
    cashflow_df["tresorerie_upper"] = cashflow_df.groupby("scenario")["yhat_upper"].cumsum() + initial_balance
    return cashflow_df


# -----------------------------
# INPUTS
# -----------------------------
left, right = st.columns([2, 1])

with right:
    st.subheader("Paramètres")
    initial_balance = st.number_input("Solde initial (trésorerie de départ)", value=0.0, step=100000.0, format="%.2f")
    horizon_days = st.slider("Horizon de prévision (jours)", min_value=30, max_value=365, value=90, step=15)
    interval_width = st.slider("Niveau d'incertitude (intervalle)", min_value=0.50, max_value=0.95, value=0.80, step=0.05)

    st.markdown("**Scénarios (appliqués sur le cash-flow prévu)**")
    opt_pct = st.slider("Optimiste: +% cash-flow", min_value=0.00, max_value=0.30, value=0.05, step=0.01)
    pess_pct = st.slider("Pessimiste: -% cash-flow", min_value=0.00, max_value=0.50, value=0.08, step=0.01)

    seuil_alerte = st.number_input("Seuil d’alerte trésorerie", value=0.0, step=100000.0, format="%.2f")

with left:
    st.subheader("Source de données (CSV)")
    mode = st.radio("Choisir la source", ["Charger un fichier CSV", "Utiliser data/transactions.csv"], horizontal=True)

    df_raw = None
    if mode == "Charger un fichier CSV":
        up = st.file_uploader("Importer transactions.csv", type=["csv"])
        if up is not None:
            df_raw = pd.read_csv(up)
    else:
        try:
            df_raw = pd.read_csv("data/transactions.csv")
        except Exception:
            st.error("Impossible de lire data/transactions.csv. Vérifie le fichier et le chemin.")
            st.stop()

    if df_raw is None:
        st.info("Charge un CSV pour afficher le dashboard.")
        st.stop()

# -----------------------------
# TRANSFORM + VALIDATION
# -----------------------------
try:
    df = prepare_transactions(df_raw)
except Exception as e:
    st.error(f"Erreur de format CSV : {e}")
    st.stop()

if df.empty:
    st.warning("Aucune donnée valide après nettoyage.")
    st.stop()

min_date, max_date = df["date"].min(), df["date"].max()

f1, f2, f3 = st.columns([1, 1, 2])
with f1:
    date_start = st.date_input("Date début", value=min_date.date(), min_value=min_date.date(), max_value=max_date.date())
with f2:
    date_end = st.date_input("Date fin", value=max_date.date(), min_value=min_date.date(), max_value=max_date.date())
with f3:
    cats = sorted(df["categorie"].unique().tolist())
    selected_cats = st.multiselect("Catégories", options=cats, default=cats)

mask = (
    (df["date"].dt.date >= date_start)
    & (df["date"].dt.date <= date_end)
    & (df["categorie"].isin(selected_cats))
)
df_f = df.loc[mask].copy()

if df_f.empty:
    st.warning("Aucune donnée sur la période / catégories sélectionnées.")
    st.stop()

# -----------------------------
# CASHFLOW DAILY + FORECAST (ML)
# -----------------------------
daily = daily_cashflow(df_f)

# Guardrail: si trop court, prévenir
if daily.shape[0] < 30:
    st.warning("Historique < 30 jours : la prévision peut être peu fiable. Idéalement 90+ jours.")

fcst = fit_forecast_ml(daily, horizon_days=horizon_days, interval_width=interval_width)
scenarios = build_scenarios(fcst, opt_pct=opt_pct, pess_pct=pess_pct)

# Trésorerie projetée (cumulative)
scen_treasury = treasury_curve_from_cashflow(scenarios, initial_balance=initial_balance)

# Réel: trésorerie basée sur cashflow réel de la période filtrée
real_treasury = daily.copy()
real_treasury["tresorerie_reelle"] = real_treasury["y"].cumsum() + initial_balance

# -----------------------------
# KPIs
# -----------------------------
solde_actuel = float(real_treasury["tresorerie_reelle"].iloc[-1])
cashflow_30d = float(daily.tail(30)["y"].sum()) if daily.shape[0] >= 30 else float(daily["y"].sum())

base_curve = scen_treasury[scen_treasury["scenario"] == "Base"].copy()
solde_fin = float(base_curve["tresorerie"].iloc[-1])
point_bas = float(base_curve["tresorerie"].min())
date_point_bas = base_curve.loc[base_curve["tresorerie"].idxmin(), "ds"]

jours_sous_seuil = int((base_curve["tresorerie"] < seuil_alerte).sum()) if seuil_alerte is not None else 0

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Trésorerie actuelle (réelle)", fmt_money(solde_actuel))
k2.metric("Trésorerie fin d’horizon (Base)", fmt_money(solde_fin))
k3.metric("Point bas (Base)", fmt_money(point_bas))
k4.metric("Date point bas (Base)", date_point_bas.strftime("%Y-%m-%d"))
k5.metric("Jours sous seuil (Base)", f"{jours_sous_seuil}")

# -----------------------------
# VISUALS
# -----------------------------
c_left, c_right = st.columns([2, 1])

with c_left:
    st.subheader("Trésorerie réelle + projetée (scénarios)")
    fig = go.Figure()

    # Réel
    fig.add_trace(go.Scatter(
        x=real_treasury["ds"], y=real_treasury["tresorerie_reelle"],
        mode="lines", name="Trésorerie réelle"
    ))

    # Scénarios trésorerie
    for sc in ["Base", "Optimiste", "Pessimiste"]:
        cur = scen_treasury[scen_treasury["scenario"] == sc]
        fig.add_trace(go.Scatter(
            x=cur["ds"], y=cur["tresorerie"],
            mode="lines", name=f"Prévision trésorerie — {sc}"
        ))

    # Bande d'incertitude pour Base
    cur = scen_treasury[scen_treasury["scenario"] == "Base"]
    fig.add_trace(go.Scatter(
        x=pd.concat([cur["ds"], cur["ds"][::-1]]),
        y=pd.concat([cur["tresorerie_upper"], cur["tresorerie_lower"][::-1]]),
        fill="toself",
        line=dict(width=0),
        name="Incertitude (Base)",
        showlegend=True
    ))

    # Seuil
    fig.add_hline(y=seuil_alerte, line_dash="dash", annotation_text="Seuil d’alerte", annotation_position="top left")

    fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

with c_right:
    st.subheader("Cash-flow net")
    hist = daily.copy()
    hist["type"] = "Réel"

    fc_base = scenarios[scenarios["scenario"] == "Base"][["ds", "yhat"]].rename(columns={"yhat": "y"})
    fc_base["type"] = "Prévu (Base)"

    cf_plot = pd.concat([hist[["ds", "y", "type"]], fc_base[["ds", "y", "type"]]], ignore_index=True)
    fig2 = px.line(cf_plot, x="ds", y="y", color="type")
    fig2.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig2, use_container_width=True)

    st.caption(f"Cash-flow net sur les 30 derniers jours (réel): {fmt_money(cashflow_30d)}")

# -----------------------------
# ANALYSES COMPLEMENTAIRES
# -----------------------------
a1, a2 = st.columns(2)

with a1:
    st.subheader("Répartition par catégorie (période filtrée)")
    by_cat = df_f.groupby("categorie", as_index=False)["montant_signe"].sum().sort_values("montant_signe")
    fig3 = px.bar(by_cat, x="montant_signe", y="categorie", orientation="h")
    fig3.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig3, use_container_width=True)

with a2:
    st.subheader("Détails des transactions (extrait)")
    st.dataframe(df_f.sort_values("date", ascending=False).head(200), use_container_width=True)

st.divider()
st.subheader("Exports")

colx1, colx2 = st.columns(2)

with colx1:
    out_prev = scenarios.copy()
    out_prev["ds"] = pd.to_datetime(out_prev["ds"]).dt.strftime("%Y-%m-%d")
    st.download_button(
        "Télécharger prévisions cash-flow (CSV)",
        data=out_prev.to_csv(index=False).encode("utf-8"),
        file_name="previsions_cashflow.csv",
        mime="text/csv"
    )

with colx2:
    out_tres = scen_treasury.copy()
    out_tres["ds"] = pd.to_datetime(out_tres["ds"]).dt.strftime("%Y-%m-%d")
    st.download_button(
        "Télécharger courbes trésorerie (CSV)",
        data=out_tres.to_csv(index=False).encode("utf-8"),
        file_name="courbes_tresorerie.csv",
        mime="text/csv"
    )


st.image("image/porte drapeau SEAHORSE.png", caption="© 2026 ", width=100)