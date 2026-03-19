"""
Module 7 — Reporter
Génère un rapport complet HTML + Markdown :
  1. Top 10 événements probabilistes
  2. Classement des variantes de stratégies
  3. Heatmap levier × rendement mensuel
  4. Equity curves + buy-hold BTC + marqueurs liquidation
  5. Résumé exécutif
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml

logger = logging.getLogger(__name__)


class Reporter:
    """Génère les rapports d'analyse et de backtest."""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.output_dir = Path("reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Stockage des données pour le rapport final
        self._events_df: pd.DataFrame | None = None
        self._results_df: pd.DataFrame | None = None
        self._kelly_df: pd.DataFrame | None = None
        self._equity_curves: dict[str, pd.Series] = {}
        self._liquidations: dict[str, list] = {}
        self._btc_benchmark: pd.Series | None = None
        self._correlation_df: pd.DataFrame | None = None

        logger.info("Reporter initialisé — output: %s/", self.output_dir)

    # ── Setters ───────────────────────────────────────────────

    def set_events(self, events_df: pd.DataFrame):
        self._events_df = events_df

    def set_results(self, results_df: pd.DataFrame):
        self._results_df = results_df

    def set_kelly(self, kelly_df: pd.DataFrame):
        self._kelly_df = kelly_df

    def set_equity_curve(self, label: str, curve: pd.Series,
                         liquidation_indices: list[int] | None = None):
        self._equity_curves[label] = curve
        if liquidation_indices:
            self._liquidations[label] = liquidation_indices

    def set_btc_benchmark(self, curve: pd.Series):
        self._btc_benchmark = curve

    def set_correlations(self, corr_df: pd.DataFrame):
        self._correlation_df = corr_df

    # ── 1. Top événements probabilistes ───────────────────────

    def top_events_table(self, events_df: pd.DataFrame | None = None,
                         top_n: int = 10) -> str:
        """Top N événements probabilistes en Markdown."""
        df = events_df if events_df is not None else self._events_df
        if df is None or df.empty:
            return "Aucun événement analysé.\n"

        # Filtrer les valides et trier par rr_ratio
        valid = df[df["valide"] == True].copy()
        if valid.empty:
            valid = df.head(top_n).copy()
        else:
            valid = valid.nlargest(top_n, "rr_ratio")

        lines = [
            "## Top {} événements probabilistes\n".format(min(top_n, len(valid))),
            "| # | Événement | N | Fréq | P(up 3j) | P(down 3j) | RR ratio | p-value |",
            "|---|-----------|---|------|----------|------------|----------|---------|",
        ]

        for i, (_, row) in enumerate(valid.iterrows(), 1):
            desc = row.get("event_desc", str(row.get("event", "")))
            if len(desc) > 50:
                desc = desc[:47] + "..."
            lines.append(
                f"| {i} | {desc} | {row['N']} | {row['freq']:.3f} "
                f"| {row['p_up_3j']:.3f} | {row['p_down_3j']:.3f} "
                f"| **{row['rr_ratio']:.2f}** | {row['p_value']:.4f} |"
            )

        return "\n".join(lines) + "\n"

    # ── 2. Classement des variantes ───────────────────────────

    def strategy_ranking_table(self, results_df: pd.DataFrame | None = None,
                               top_n: int = 20) -> str:
        """Classement des variantes par Sharpe en Markdown."""
        df = results_df if results_df is not None else self._results_df
        if df is None or df.empty:
            return "Aucun résultat de backtest.\n"

        top = df.nlargest(top_n, "sharpe_ratio")

        lines = [
            "## Classement des variantes (Top {})\n".format(min(top_n, len(top))),
            "| # | Stratégie | Asset | Lev | TF | Sharpe | CAGR% | MaxDD% | Trades | Liqs | Frais% | Kelly% |",
            "|---|-----------|-------|-----|----|---------:|------:|-------:|-------:|-----:|-------:|-------:|",
        ]

        for i, (_, row) in enumerate(top.iterrows(), 1):
            kelly = row.get("kelly_fraction", 0)
            lines.append(
                f"| {i} | {row['strategy']} | {row['asset']} "
                f"| {row['leverage']:.0f}x | {row['timeframe']} "
                f"| **{row['sharpe_ratio']:.3f}** | {row['cagr']:.1f} "
                f"| {row['max_drawdown']:.1f} | {row['nb_trades']} "
                f"| {row['nb_liquidations']} | {row['fees_total']:.2f} "
                f"| {kelly:.2f} |"
            )

        return "\n".join(lines) + "\n"

    # ── 3. Heatmap levier × rendement ─────────────────────────

    def leverage_heatmap(self, results_df: pd.DataFrame | None = None
                         ) -> dict[str, go.Figure]:
        """Heatmaps levier × asset colorées par rendement mensuel moyen.
        Une figure par stratégie."""
        df = results_df if results_df is not None else self._results_df
        if df is None or df.empty:
            return {}

        figures = {}

        for strat_name in df["strategy"].unique():
            strat_df = df[df["strategy"] == strat_name]

            if strat_df.empty:
                continue

            # Calculer le rendement mensuel moyen
            strat_df = strat_df.copy()
            # Estimation : total_return / (durée en mois)
            period_years = (
                pd.Timestamp(self.cfg["period_end"]) -
                pd.Timestamp(self.cfg["period_start"])
            ).days / 365.25
            period_months = period_years * 12 * self.cfg["train_ratio"]
            strat_df["monthly_return"] = strat_df["total_return"] / max(period_months, 1)

            # Pivot : levier × asset
            pivot = strat_df.pivot_table(
                values="monthly_return",
                index="leverage",
                columns="asset",
                aggfunc="mean"
            ).sort_index()

            if pivot.empty:
                continue

            fig = go.Figure(data=go.Heatmap(
                z=pivot.values,
                x=pivot.columns.tolist(),
                y=[f"{int(v)}x" for v in pivot.index],
                colorscale=[
                    [0, "#d32f2f"],      # rouge (perte)
                    [0.4, "#ffeb3b"],    # jaune (neutre)
                    [0.6, "#4caf50"],    # vert clair
                    [1, "#1b5e20"],      # vert foncé (gain)
                ],
                zmid=0,
                text=[[f"{v:.2f}%" for v in row] for row in pivot.values],
                texttemplate="%{text}",
                textfont={"size": 14},
                colorbar={"title": "Rend. mens. %"},
            ))

            fig.update_layout(
                title=f"Heatmap — {strat_name.upper()} — Rendement mensuel moyen par levier × asset",
                xaxis_title="Asset",
                yaxis_title="Levier",
                height=400,
                width=700,
                template="plotly_dark",
            )

            figures[strat_name] = fig

        return figures

    # ── 4. Equity curves ──────────────────────────────────────

    def equity_curves_figure(self,
                             curves: dict[str, pd.Series] | None = None,
                             liquidations: dict[str, list] | None = None,
                             btc_benchmark: pd.Series | None = None
                             ) -> go.Figure:
        """Equity curves superposées + buy-hold BTC + marqueurs liquidation."""
        if curves is None:
            curves = self._equity_curves
        if liquidations is None:
            liquidations = self._liquidations
        if btc_benchmark is None:
            btc_benchmark = self._btc_benchmark

        if not curves:
            return go.Figure()

        fig = go.Figure()

        # Palette de couleurs
        colors = [
            "#2196f3", "#4caf50", "#ff9800", "#e91e63",
            "#9c27b0", "#00bcd4", "#ff5722", "#8bc34a",
            "#3f51b5", "#ffc107", "#607d8b", "#795548",
        ]

        for i, (label, curve) in enumerate(curves.items()):
            color = colors[i % len(colors)]

            fig.add_trace(go.Scatter(
                x=curve.index,
                y=curve.values,
                name=label,
                line={"color": color, "width": 1.5},
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}%<extra>" + label + "</extra>",
            ))

            # Marqueurs de liquidation
            liq_idx = liquidations.get(label, [])
            if liq_idx:
                liq_times = [curve.index[j] for j in liq_idx if j < len(curve)]
                liq_vals = [curve.iloc[j] for j in liq_idx if j < len(curve)]
                fig.add_trace(go.Scatter(
                    x=liq_times,
                    y=liq_vals,
                    mode="markers",
                    name=f"{label} — liquidations",
                    marker={
                        "symbol": "x",
                        "size": 12,
                        "color": "#d32f2f",
                        "line": {"width": 2, "color": "#fff"},
                    },
                    showlegend=True,
                    hovertemplate="LIQUIDATION<br>%{x|%Y-%m-%d}<br>%{y:.2f}%<extra></extra>",
                ))

        # Buy & Hold BTC
        if btc_benchmark is not None and not btc_benchmark.empty:
            fig.add_trace(go.Scatter(
                x=btc_benchmark.index,
                y=btc_benchmark.values,
                name="Buy & Hold BTC",
                line={"color": "#ffd700", "width": 2, "dash": "dot"},
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}%<extra>BTC B&H</extra>",
            ))

        # Ligne de départ (100%)
        fig.add_hline(
            y=100, line_dash="dash", line_color="gray",
            annotation_text="Capital initial (100%)",
        )

        # Ligne de stop global
        stop = self.cfg.get("stop_global_portfolio", 50)
        fig.add_hline(
            y=stop, line_dash="dash", line_color="#d32f2f",
            annotation_text=f"Stop global ({stop}%)",
        )

        fig.update_layout(
            title="Equity Curves — Stratégies vs Buy & Hold BTC",
            xaxis_title="Date",
            yaxis_title="Portfolio (%)",
            height=600,
            width=1100,
            template="plotly_dark",
            legend={"orientation": "h", "y": -0.15},
            hovermode="x unified",
        )

        return fig

    # ── 5. Résumé exécutif ────────────────────────────────────

    def executive_summary(self, results_df: pd.DataFrame | None = None,
                          top_n: int = 5) -> str:
        """Top 5 configurations par Sharpe avec note de risque 1–5."""
        df = results_df if results_df is not None else self._results_df
        if df is None or df.empty:
            return "Aucun résultat.\n"

        top = df.nlargest(top_n, "sharpe_ratio")

        lines = [
            "## Résumé exécutif — Top {} configurations\n".format(min(top_n, len(top))),
            "| # | Stratégie | Asset | Config | Sharpe | Return% | MaxDD% | Risque |",
            "|---|-----------|-------|--------|-------:|--------:|-------:|:------:|",
        ]

        for i, (_, row) in enumerate(top.iterrows(), 1):
            config = f"{row['leverage']:.0f}x / {row['size_pct']:.0f}%"
            risk = self._risk_score(row)
            risk_emoji = self._risk_label(risk)

            lines.append(
                f"| {i} | **{row['strategy']}** | {row['asset']} "
                f"| {config} | **{row['sharpe_ratio']:.3f}** "
                f"| {row['total_return']:.1f} | {row['max_drawdown']:.1f} "
                f"| {risk_emoji} |"
            )

        lines.append("")
        lines.append("*Risque: 1/5=faible, 2/5=modéré, 3/5=élevé, 4/5=très élevé, 5/5=extrême*")

        return "\n".join(lines) + "\n"

    @staticmethod
    def _risk_score(row: pd.Series) -> int:
        """Note de risque 1–5 basée sur nb_liquidations et max_drawdown."""
        score = 1

        # Liquidations
        liqs = row.get("nb_liquidations", 0)
        if liqs >= 5:
            score += 2
        elif liqs >= 2:
            score += 1

        # Max drawdown
        dd = row.get("max_drawdown", 0)
        if dd >= 40:
            score += 2
        elif dd >= 25:
            score += 1

        # Levier
        lev = row.get("leverage", 1)
        if lev >= 10:
            score += 1

        return min(score, 5)

    @staticmethod
    def _risk_label(score: int) -> str:
        labels = {1: "1/5", 2: "2/5", 3: "3/5", 4: "4/5", 5: "5/5"}
        return labels.get(score, "?/5")

    # ── Corrélation laggée ────────────────────────────────────

    def correlation_table(self, corr_df: pd.DataFrame | None = None) -> str:
        """Table des corrélations laggées multi-asset."""
        df = corr_df if corr_df is not None else self._correlation_df
        if df is None or df.empty:
            return "Aucune analyse de corrélation.\n"

        lines = [
            "## Corrélations laggées multi-asset\n",
            "| Paire | Lag optimal | Corrélation | Direction (leader → follower) |",
            "|-------|:----------:|:-----------:|-------------------------------|",
        ]

        for _, row in df.iterrows():
            lines.append(
                f"| {row['asset_A']} / {row['asset_B']} "
                f"| {row['optimal_lag']}h "
                f"| {row['correlation']:.4f} "
                f"| {row['direction']} |"
            )

        return "\n".join(lines) + "\n"

    # ── Rapport complet ───────────────────────────────────────

    def generate_report(self, output_dir: str | None = None) -> str:
        """Génère le rapport complet en HTML + Markdown.
        Retourne le chemin du fichier HTML."""

        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # ── Markdown ──────────────────────────────────────────
        md_parts = [
            f"# Crypto Strategy Research Report",
            f"*Généré le {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n",
            f"**Période**: {self.cfg['period_start']} → {self.cfg['period_end']}  ",
            f"**Assets**: {', '.join(self.cfg['assets'])}  ",
            f"**Capital**: 100% (normalisé)  ",
            f"**Stop global**: {self.cfg['stop_global_portfolio']}%\n",
            "---\n",
        ]

        # 1. Événements probabilistes
        md_parts.append(self.top_events_table())
        md_parts.append("")

        # Corrélations
        md_parts.append(self.correlation_table())
        md_parts.append("")

        # 2. Classement des variantes
        md_parts.append(self.strategy_ranking_table())
        md_parts.append("")

        # 5. Résumé exécutif
        md_parts.append(self.executive_summary())
        md_parts.append("")

        # Kelly
        if self._kelly_df is not None and not self._kelly_df.empty:
            md_parts.append("## Kelly Fractional — Résultats\n")
            md_parts.append(
                "| Stratégie | Asset | Kelly% | Sharpe (Kelly) | Sharpe (train) | Return% (Kelly) |"
            )
            md_parts.append(
                "|-----------|-------|-------:|---------------:|---------------:|----------------:|"
            )
            for _, row in self._kelly_df.iterrows():
                md_parts.append(
                    f"| {row['strategy']} | {row['asset']} "
                    f"| {row['kelly_fraction']:.2f} "
                    f"| **{row['sharpe_kelly']:.3f}** "
                    f"| {row['sharpe_train']:.3f} "
                    f"| {row['total_return_kelly']:.1f} |"
                )
            md_parts.append("")

        md_content = "\n".join(md_parts)

        # Sauver le Markdown
        md_path = self.output_dir / f"report_{timestamp}.md"
        md_path.write_text(md_content, encoding="utf-8")
        logger.info("Rapport Markdown: %s", md_path)

        # ── HTML ──────────────────────────────────────────────
        html_parts = [
            "<!DOCTYPE html>",
            '<html lang="fr">',
            "<head>",
            '  <meta charset="UTF-8">',
            '  <meta name="viewport" content="width=device-width, initial-scale=1.0">',
            "  <title>Crypto Strategy Research Report</title>",
            '  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>',
            "  <style>",
            "    :root { --bg: #1a1a2e; --surface: #16213e; --text: #e0e0e0; ",
            "            --accent: #0f3460; --highlight: #e94560; }",
            "    * { margin: 0; padding: 0; box-sizing: border-box; }",
            "    body { background: var(--bg); color: var(--text); ",
            "           font-family: 'SF Mono', 'Fira Code', monospace; ",
            "           padding: 2rem; line-height: 1.6; }",
            "    h1 { color: #fff; margin-bottom: 0.5rem; font-size: 1.8rem; }",
            "    h2 { color: var(--highlight); margin: 2rem 0 1rem; font-size: 1.3rem; }",
            "    .meta { color: #888; margin-bottom: 2rem; }",
            "    table { border-collapse: collapse; width: 100%; margin: 1rem 0; ",
            "            background: var(--surface); border-radius: 8px; overflow: hidden; }",
            "    th { background: var(--accent); padding: 0.7rem 1rem; ",
            "         text-align: left; font-size: 0.85rem; color: #aaa; }",
            "    td { padding: 0.5rem 1rem; border-bottom: 1px solid #ffffff10; ",
            "         font-size: 0.85rem; }",
            "    tr:hover { background: #ffffff08; }",
            "    .positive { color: #4caf50; }",
            "    .negative { color: #f44336; }",
            "    .chart-container { margin: 2rem 0; background: var(--surface); ",
            "                       border-radius: 8px; padding: 1rem; }",
            "    .risk-1 { color: #4caf50; } .risk-2 { color: #8bc34a; }",
            "    .risk-3 { color: #ffeb3b; } .risk-4 { color: #ff9800; }",
            "    .risk-5 { color: #f44336; font-weight: bold; }",
            "    .summary-box { background: var(--surface); border-radius: 8px; ",
            "                   padding: 1.5rem; margin: 1rem 0; ",
            "                   border-left: 4px solid var(--highlight); }",
            "  </style>",
            "</head>",
            "<body>",
            "  <h1>Crypto Strategy Research Report</h1>",
            f'  <div class="meta">Généré le {datetime.now().strftime("%Y-%m-%d %H:%M")} &mdash; ',
            f'    Période: {self.cfg["period_start"]} &rarr; {self.cfg["period_end"]} &mdash; ',
            f'    Assets: {", ".join(self.cfg["assets"])}</div>',
        ]

        # Tables HTML
        html_parts.append(self._events_to_html())
        html_parts.append(self._ranking_to_html())
        html_parts.append(self._summary_to_html())

        if self._kelly_df is not None and not self._kelly_df.empty:
            html_parts.append(self._kelly_to_html())

        # Charts plotly
        chart_idx = 0

        # Heatmaps
        heatmaps = self.leverage_heatmap()
        for strat_name, fig in heatmaps.items():
            div_id = f"heatmap_{chart_idx}"
            html_parts.append(f'  <div class="chart-container" id="{div_id}"></div>')
            html_parts.append("  <script>")
            html_parts.append(f"    Plotly.newPlot('{div_id}', "
                              f"{fig.to_json()}.data, "
                              f"{fig.to_json()}.layout);")
            html_parts.append("  </script>")
            chart_idx += 1

        # Equity curves
        eq_fig = self.equity_curves_figure()
        if eq_fig.data:
            div_id = f"equity_{chart_idx}"
            html_parts.append(f'  <div class="chart-container" id="{div_id}"></div>')
            html_parts.append("  <script>")
            fig_json = eq_fig.to_json()
            html_parts.append(f"    var figData = {fig_json};")
            html_parts.append(f"    Plotly.newPlot('{div_id}', figData.data, figData.layout);")
            html_parts.append("  </script>")

        html_parts.extend([
            "</body>",
            "</html>",
        ])

        html_content = "\n".join(html_parts)
        html_path = self.output_dir / f"report_{timestamp}.html"
        html_path.write_text(html_content, encoding="utf-8")
        logger.info("Rapport HTML: %s", html_path)

        return str(html_path)

    # ── HTML helpers ──────────────────────────────────────────

    def _events_to_html(self) -> str:
        df = self._events_df
        if df is None or df.empty:
            return ""

        valid = df[df["valide"] == True].nlargest(10, "rr_ratio") if "valide" in df.columns else df.head(10)
        if valid.empty:
            valid = df.head(10)

        rows = []
        for i, (_, r) in enumerate(valid.iterrows(), 1):
            desc = r.get("event_desc", "")[:50]
            rr_class = "positive" if r["rr_ratio"] > 1.5 else ""
            rows.append(
                f'    <tr><td>{i}</td><td>{desc}</td><td>{r["N"]}</td>'
                f'<td>{r["freq"]:.3f}</td><td>{r["p_up_3j"]:.3f}</td>'
                f'<td>{r["p_down_3j"]:.3f}</td>'
                f'<td class="{rr_class}"><strong>{r["rr_ratio"]:.2f}</strong></td>'
                f'<td>{r["p_value"]:.4f}</td></tr>'
            )

        return (
            '  <h2>Top 10 événements probabilistes</h2>\n'
            '  <table>\n'
            '    <tr><th>#</th><th>Événement</th><th>N</th><th>Fréq</th>'
            '<th>P(up 3j)</th><th>P(down 3j)</th><th>RR ratio</th><th>p-value</th></tr>\n'
            + "\n".join(rows) +
            '\n  </table>'
        )

    def _ranking_to_html(self) -> str:
        df = self._results_df
        if df is None or df.empty:
            return ""

        top = df.nlargest(20, "sharpe_ratio")
        rows = []
        for i, (_, r) in enumerate(top.iterrows(), 1):
            ret_class = "positive" if r["total_return"] > 0 else "negative"
            rows.append(
                f'    <tr><td>{i}</td><td>{r["strategy"]}</td><td>{r["asset"]}</td>'
                f'<td>{r["leverage"]:.0f}x</td><td>{r["timeframe"]}</td>'
                f'<td><strong>{r["sharpe_ratio"]:.3f}</strong></td>'
                f'<td class="{ret_class}">{r["cagr"]:.1f}%</td>'
                f'<td class="negative">{r["max_drawdown"]:.1f}%</td>'
                f'<td>{r["nb_trades"]}</td><td>{r["nb_liquidations"]}</td>'
                f'<td>{r["fees_total"]:.2f}%</td>'
                f'<td>{r.get("kelly_fraction", 0):.2f}%</td></tr>'
            )

        return (
            '  <h2>Classement des variantes</h2>\n'
            '  <table>\n'
            '    <tr><th>#</th><th>Strat</th><th>Asset</th><th>Lev</th><th>TF</th>'
            '<th>Sharpe</th><th>CAGR</th><th>MaxDD</th><th>Trades</th><th>Liqs</th>'
            '<th>Frais</th><th>Kelly</th></tr>\n'
            + "\n".join(rows) +
            '\n  </table>'
        )

    def _summary_to_html(self) -> str:
        df = self._results_df
        if df is None or df.empty:
            return ""

        top = df.nlargest(5, "sharpe_ratio")
        rows = []
        for i, (_, r) in enumerate(top.iterrows(), 1):
            risk = self._risk_score(r)
            config = f"{r['leverage']:.0f}x / {r['size_pct']:.0f}%"
            rows.append(
                f'    <tr><td>{i}</td><td><strong>{r["strategy"]}</strong></td>'
                f'<td>{r["asset"]}</td><td>{config}</td>'
                f'<td><strong>{r["sharpe_ratio"]:.3f}</strong></td>'
                f'<td>{r["total_return"]:.1f}%</td>'
                f'<td>{r["max_drawdown"]:.1f}%</td>'
                f'<td class="risk-{risk}">{risk}/5</td></tr>'
            )

        return (
            '  <h2>Résumé exécutif — Top 5</h2>\n'
            '  <div class="summary-box">\n'
            '  <table>\n'
            '    <tr><th>#</th><th>Stratégie</th><th>Asset</th><th>Config</th>'
            '<th>Sharpe</th><th>Return</th><th>MaxDD</th><th>Risque</th></tr>\n'
            + "\n".join(rows) +
            '\n  </table>\n  </div>'
        )

    def _kelly_to_html(self) -> str:
        df = self._kelly_df
        if df is None or df.empty:
            return ""

        rows = []
        for _, r in df.iterrows():
            rows.append(
                f'    <tr><td>{r["strategy"]}</td><td>{r["asset"]}</td>'
                f'<td>{r["kelly_fraction"]:.2f}%</td>'
                f'<td><strong>{r["sharpe_kelly"]:.3f}</strong></td>'
                f'<td>{r["sharpe_train"]:.3f}</td>'
                f'<td>{r["total_return_kelly"]:.1f}%</td></tr>'
            )

        return (
            '  <h2>Kelly Fractional</h2>\n'
            '  <table>\n'
            '    <tr><th>Strat</th><th>Asset</th><th>Kelly%</th>'
            '<th>Sharpe (Kelly)</th><th>Sharpe (train)</th><th>Return (Kelly)</th></tr>\n'
            + "\n".join(rows) +
            '\n  </table>'
        )


# ── Standalone test ───────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-20s | %(levelname)-5s | %(message)s"
    )

    reporter = Reporter()

    # Données factices
    events = pd.DataFrame([
        {"event_desc": "RSI<25 & BB<0 & vol>1.5", "N": 45, "freq": 0.02,
         "p_up_3j": 0.65, "p_down_3j": 0.18, "rr_ratio": 3.61,
         "p_value": 0.001, "valide": True},
        {"event_desc": "golden_cross & bull & RSI 50-65", "N": 38, "freq": 0.015,
         "p_up_3j": 0.55, "p_down_3j": 0.25, "rr_ratio": 2.20,
         "p_value": 0.03, "valide": True},
        {"event_desc": "compression & breakout & vol>3", "N": 22, "freq": 0.008,
         "p_up_3j": 0.72, "p_down_3j": 0.15, "rr_ratio": 4.80,
         "p_value": 0.002, "valide": False},
    ])
    reporter.set_events(events)

    results = pd.DataFrame([
        {"strategy": "momentum", "asset": "BTC/USDT", "timeframe": "4h",
         "leverage": 5, "size_pct": 20, "sl_pct": 2, "tp_pct": 6,
         "sharpe_ratio": 1.85, "cagr": 45.2, "total_return": 85.3,
         "max_drawdown": 18.5, "nb_trades": 42, "nb_liquidations": 1,
         "fees_total": 3.2, "kelly_fraction": 4.5, "phase": "test",
         "win_rate": 0.55, "profit_factor": 1.8, "avg_win": 3.2, "avg_loss": 1.8,
         "sortino_ratio": 2.1, "funding_cost_total": 1.5, "avg_duration_hours": 36,
         "final_capital": 185.3, "max_dd_duration_days": 12},
        {"strategy": "breakout", "asset": "ETH/USDT", "timeframe": "1h",
         "leverage": 10, "size_pct": 5, "sl_pct": 1.5, "tp_pct": 5,
         "sharpe_ratio": 1.42, "cagr": 32.1, "total_return": 52.8,
         "max_drawdown": 28.3, "nb_trades": 65, "nb_liquidations": 4,
         "fees_total": 2.1, "kelly_fraction": 3.2, "phase": "test",
         "win_rate": 0.48, "profit_factor": 1.5, "avg_win": 4.5, "avg_loss": 2.1,
         "sortino_ratio": 1.6, "funding_cost_total": 0.8, "avg_duration_hours": 12,
         "final_capital": 152.8, "max_dd_duration_days": 25},
    ])
    reporter.set_results(results)

    # Générer
    print(reporter.top_events_table())
    print(reporter.strategy_ranking_table())
    print(reporter.executive_summary())

    path = reporter.generate_report()
    print(f"\nRapport généré: {path}")
