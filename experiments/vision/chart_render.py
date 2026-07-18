"""Candlestick chart renderer for the vision structure agent.

Renders OHLCV DataFrames as PNG: candles + EMA50/EMA200 + volume subplot +
optional horizontal level lines. Lives in experiments/ until the vision gates pass and it
is promoted into src/ (matplotlib would become a prod dependency).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

UP = "#26a69a"
DOWN = "#ef5350"


def render_chart(
    df,
    out_path: str,
    title: str,
    show_bars: int = 120,
    levels: dict | None = None,
) -> str:
    """df: DataFrame with open/high/low/close/volume and a datetime index.
    Extra history beyond show_bars improves EMA accuracy; pass what you have.
    levels: {"label": price} drawn as dashed horizontal lines."""
    d = df.copy()
    d["ema50"] = d["close"].ewm(span=50, adjust=False).mean()
    d["ema200"] = d["close"].ewm(span=200, adjust=False).mean()
    d = d.tail(show_bars)

    fig, (ax, axv) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True,
        gridspec_kw={"height_ratios": [4, 1]}, dpi=110,
    )
    for i, (_, r) in enumerate(d.iterrows()):
        c = UP if r["close"] >= r["open"] else DOWN
        ax.plot([i, i], [r["low"], r["high"]], color=c, linewidth=0.8, zorder=2)
        body = abs(r["close"] - r["open"]) or (r["high"] - r["low"]) * 0.02
        ax.add_patch(plt.Rectangle(
            (i - 0.35, min(r["open"], r["close"])), 0.7, body,
            facecolor=c, edgecolor=c, zorder=3,
        ))
        axv.bar(i, r["volume"], color=c, width=0.7)

    x = range(len(d))
    ax.plot(x, d["ema50"].values, color="#2962ff", linewidth=1.3, label="EMA 50")
    ax.plot(x, d["ema200"].values, color="#f57f17", linewidth=1.3, label="EMA 200")
    for label, price in (levels or {}).items():
        if price:
            ax.axhline(price, color="#616161", linestyle="--", linewidth=1)
            ax.annotate(label, (0, price), fontsize=8, color="#616161",
                        va="bottom")

    step = max(1, len(d) // 8)
    ticks = list(range(0, len(d), step))
    ax.set_xticks(ticks)
    ax.set_xticklabels(
        [str(d.index[t])[:16] for t in ticks], rotation=25, fontsize=8
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.25)
    axv.set_ylabel("vol", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
