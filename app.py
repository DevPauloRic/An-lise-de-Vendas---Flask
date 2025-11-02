from flask import Flask, render_template
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio

app = Flask(__name__)

# --- Função de insights automáticos ---
def auto_insights(trend, bars, corr):
    insights = []
    vals = trend["values"]
    if len(vals) >= 4:
        recent = np.mean(vals[-3:])
        prior = np.mean(vals[:-3]) if len(vals) > 3 else np.mean(vals)
        delta = (recent - prior) / prior * 100 if prior else 0
        if abs(delta) >= 5:
            insights.append(f"Receita recente está {'acelerando' if delta>0 else 'desacelerando'} {delta:.1f}% vs média anterior.")
    if bars["values"]:
        i_max = int(np.argmax(bars["values"]))
        i_min = int(np.argmin(bars["values"]))
        insights.append(f"Canal líder: {bars['labels'][i_max]} (R$ {bars['values'][i_max]:,.2f}); atenção ao canal {bars['labels'][i_min]}.")
    r = corr["r"]
    if abs(r) >= 0.5:
        force = "forte" if abs(r) >= 0.7 else "moderada"
        sentido = "positiva" if r > 0 else "negativa"
        insights.append(f"Correlação {force} {sentido} entre gastos em anúncios e receita (r={r:.2f}).")
    if corr["outliers"]:
        outs = [corr["labels"][i] for i in corr["outliers"]]
        insights.append(f"Outliers detectados: {', '.join(outs)}.")
    return insights

@app.route('/')
def index():
    # --- 1. Ler e preparar dados ---
    df = pd.read_csv("data.csv")
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.strftime("%b/%Y")
    df["ads_spend"] = pd.to_numeric(df["ads_spend"], errors="coerce")

    # --- 2. Gráfico de tendência mensal ---
    trend_df = df.groupby("month")["revenue"].sum().reset_index()
    trend_fig = px.line(trend_df, x="month", y="revenue", title="Tendência Mensal de Receita", markers=True)
    trend_html = pio.to_html(trend_fig, full_html=False)

    # --- 3. Comparação de categorias ---
    bars_df = df.groupby("channel")["revenue"].mean().reset_index()
    bars_fig = px.bar(bars_df, x="channel", y="revenue", title="Receita Média por Canal", color="channel")
    bars_html = pio.to_html(bars_fig, full_html=False)

        # --- 4. Correlação e outliers ---
    # Converte colunas para numéricas e remove linhas com NaN ou infinitos
    df["ads_spend"] = pd.to_numeric(df["ads_spend"], errors="coerce")
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["ads_spend", "revenue"])

    corr_val = df["ads_spend"].corr(df["revenue"])

    corr_fig = px.scatter(
        df,
        x="ads_spend",
        y="revenue",
        color="channel",
        title=f"Correlação entre Gastos e Receita (r={corr_val:.2f})"
    )
    corr_html = pio.to_html(corr_fig, full_html=False)


    # Detectar outliers (simplificado)
    resid = df["revenue"] - np.poly1d(np.polyfit(df["ads_spend"], df["revenue"], 1))(df["ads_spend"])
    outliers_idx = np.where(abs(resid) > np.std(resid) * 1.5)[0]

    # --- 5. Gerar insights automáticos ---
    trend = {"labels": trend_df["month"].tolist(), "values": trend_df["revenue"].tolist()}
    bars = {"labels": bars_df["channel"].tolist(), "values": bars_df["revenue"].tolist()}
    corr = {"labels": df["month"].tolist(), "r": corr_val, "outliers": outliers_idx.tolist()}

    insights = auto_insights(trend, bars, corr)

    return render_template("index.html",
                           trend_html=trend_html,
                           bars_html=bars_html,
                           corr_html=corr_html,
                           insights=insights)

if __name__ == "__main__":
    app.run(debug=True)
