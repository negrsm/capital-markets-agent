# AI Capital Markets Decision Agent

> A multi-agent AI system for sector ETF analysis and portfolio recommendation generation.

---

## Overview

The **AI Capital Markets Decision Agent** is an end-to-end quantitative research pipeline that combines classical financial analysis, machine learning, and large language models to produce actionable portfolio recommendations across a universe of sector ETFs. The system ingests daily OHLCV market data, computes institutional-grade performance metrics (annualized return, Sharpe ratio, maximum drawdown, moving-average signals), trains per-ticker Random Forest classifiers to forecast next-week return direction, and integrates OpenAI GPT-4o-mini to overlay an analyst-style sentiment layer. A seven-factor scoring rubric then converts these signals into explicit portfolio actions — Overweight, Neutral, Underweight, or Avoid — accompanied by concentration and volatility risk flags and an AI-generated executive summary. The full pipeline is orchestrated by LangGraph and surfaced through an interactive Streamlit dashboard styled to institutional standards.

---

## Architecture

Seven specialized agents run in a strict linear sequence. Each agent reads the shared pipeline state, performs its task, and writes its outputs back before the next agent begins.

```
┌─────────────────────────────────────────────────────────────────────┐
│                   AI Capital Markets Decision Agent                  │
│                        LangGraph Pipeline                            │
└─────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
  │  1. Market   │     │  2. Metrics  │     │  3. Forecasting  │
  │  Data Agent  │────▶│    Agent     │────▶│      Agent       │
  │              │     │              │     │                  │
  │ yfinance CSV │     │ Sharpe, Vol, │     │ Random Forest    │
  │ OHLCV data   │     │ Drawdown, MA │     │ P(up next week)  │
  └──────────────┘     └──────────────┘     └──────────────────┘
                                                      │
         ┌────────────────────────────────────────────┘
         ▼
  ┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
  │  4. Sentiment│     │  5. Decision │     │   6. Risk Agent  │
  │    Agent     │────▶│    Agent     │────▶│                  │
  │              │     │              │     │ Concentration &  │
  │ GPT-4o-mini  │     │ 7-factor     │     │ Volatility flags │
  │ Bullish /    │     │ scoring      │     │                  │
  │ Neutral /    │     │ rubric       │     │                  │
  │ Bearish      │     │              │     │                  │
  └──────────────┘     └──────────────┘     └──────────────────┘
                                                      │
         ┌────────────────────────────────────────────┘
         ▼
  ┌──────────────┐     ┌──────────────────────────────────────┐
  │  7. Report   │     │         Streamlit Dashboard          │
  │    Agent     │────▶│                                      │
  │              │     │  Metric cards · Recommendations      │
  │ GPT-4o-mini  │     │  Forecast chart · Heatmap            │
  │ 200-word     │     │  Cumulative returns · Risk warnings  │
  │ executive    │     │  Executive summary                   │
  │ summary      │     │                                      │
  └──────────────┘     └──────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Agent orchestration | LangGraph |
| LLM framework | LangChain |
| Language model | OpenAI GPT-4o-mini |
| ML forecasting | scikit-learn — Random Forest Classifier |
| Market data | yfinance |
| Dashboard | Streamlit |
| Data manipulation | pandas |
| Visualization | seaborn, matplotlib, Altair |

---

## Project Structure

```
capital-markets-agent/
│
├── app/
│   └── dashboard.py          # Streamlit dashboard (entry point for UI)
│
├── data/
│   ├── raw/                  # Downloaded OHLCV CSVs (one per ticker)
│   │   ├── SPY.csv
│   │   ├── QQQ.csv
│   │   ├── XLF.csv
│   │   ├── XLE.csv
│   │   ├── XLV.csv
│   │   ├── TLT.csv
│   │   ├── GLD.csv
│   │   └── UUP.csv
│   └── processed/            # Pipeline outputs (auto-created on first run)
│       ├── forecasting_probabilities.csv
│       └── forecasting_model_report.txt
│
├── notebooks/
│   └── 02_financial_metrics.ipynb
│
├── src/
│   ├── data_loader.py        # Downloads and loads OHLCV data via yfinance
│   ├── metrics.py            # Computes Sharpe, drawdown, volatility, beta, MAs
│   ├── forecasting.py        # Builds features, trains Random Forest, saves results
│   ├── agents.py             # Defines MarketState + all 7 LangGraph agent nodes
│   ├── graph.py              # Wires the graph; exposes run_pipeline()
│   └── report_generator.py
│
├── .env                      # OPENAI_API_KEY (not committed)
├── .gitignore
├── main.py
├── requirements.txt
└── README.md
```

---

## Key Results

Sample output from a full pipeline run (data as of May 2025):

| Ticker | Name | Recommendation | Score | Sentiment | Forecast Prob |
|--------|------|---------------|-------|-----------|--------------|
| GLD | Gold ETF | **Overweight** | 6/7 | Bullish | 67% |
| SPY | S&P 500 | **Overweight** | 6/7 | Bullish | 62% |
| QQQ | Nasdaq 100 | Neutral | 5/7 | Neutral | 58% |
| XLV | Healthcare | Neutral | 4/7 | Neutral | 52% |
| XLE | Energy | Neutral | 4/7 | Neutral | 49% |
| UUP | USD Index | Underweight | 3/7 | Bearish | 43% |
| TLT | Long-Term Bonds | **Underweight** | 2/7 | Bearish | 38% |
| XLF | Financials | **Avoid** | 1/7 | Bearish | 31% |

**Risk flags raised:**
- No concentration risk (≤ 3 Overweight positions)
- TLT flagged for elevated realised volatility (> 30% annualised)

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/negrsm/capital-markets-agent.git
cd capital-markets-agent
```

### 2. Create a conda environment

```bash
conda create -n capital-markets python=3.11 -y
conda activate capital-markets
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your OpenAI API key

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...your-key-here...
```

### 5. Download market data

```bash
python src/data_loader.py
```

This downloads daily OHLCV data for all 8 tickers from 2022-01-01 to today and saves CSVs to `data/raw/`.

### 6. Run the full pipeline (CLI)

```bash
python src/graph.py
```

Trains all models, calls the LLM agents, and prints the decisions and executive summary to the terminal.

### 7. Launch the interactive dashboard

```bash
streamlit run app/dashboard.py
```

Open `http://localhost:8501`, then click **Run Full Pipeline** in the sidebar to execute the full agent chain and explore the results interactively.

---

## Methodology

### OHLCV Data Collection
Daily Open/High/Low/Close/Volume data is downloaded via `yfinance` for 8 sector ETFs spanning equities (SPY, QQQ), sector funds (XLF, XLE, XLV), bonds (TLT), commodities (GLD), and currencies (UUP). Data is cached locally as CSVs to avoid redundant API calls.

### Financial Metrics Calculation
`metrics.py` computes institutional performance metrics for each ticker:
- **Annualized return** — geometric compounding over 252 trading days
- **Annualized volatility** — rolling standard deviation scaled by √252
- **Sharpe ratio** — excess return over the 5% risk-free rate divided by volatility
- **Maximum drawdown** — largest peak-to-trough decline in price history
- **50-day and 200-day moving averages** — trend signals
- **Beta** — covariance with SPY as the market benchmark

### ML Forecasting with Chronological Train/Test Split
`forecasting.py` constructs 9 features per ticker (lag returns at 1d/5d/20d, realized volatility, momentum, MA signals, volume change) and trains a `RandomForestClassifier` to predict whether the next 5-day return will be positive. Critically, the data is split **chronologically** (80% train / 20% test, no shuffling) to replicate how a live model would be evaluated — the test set is always a future period the model has never seen. This eliminates data leakage that random shuffling would introduce in time-series settings.

### LangGraph Agent Orchestration
The pipeline is built as a LangGraph `StateGraph` with a single shared `MarketState` TypedDict. Each of the 7 agent nodes reads from and writes to this shared state. LangGraph handles the execution order, state merging, and error boundaries, making it straightforward to add conditional routing or parallel branches in future iterations.

### LLM Sentiment Classification
`sentiment_agent` sends a single batched prompt to GPT-4o-mini containing quantitative summaries of all 8 tickers. The model returns a structured JSON object mapping each ticker to a sentiment label (Bullish / Neutral / Bearish). Batching all tickers into one API call reduces latency and cost compared to one call per ticker.

### Decision Scoring Rubric
`decision_agent` applies a transparent 7-factor binary scoring model. Each factor is an independently verifiable quantitative or signal-based criterion:

```
+1  Annualized return > 10%
+1  Annualized volatility < 20%
+1  Sharpe ratio > 0.5
+1  Price above 200-day moving average
+1  ML forecast probability > 55%
+1  LLM sentiment = Bullish
+1  Maximum drawdown > -25%
─────────────────────────────
Score 6–7  →  Overweight
Score 4–5  →  Neutral
Score 2–3  →  Underweight
Score 0–1  →  Avoid
```

---

## Business Context

This project is directly relevant to **capital markets portfolio management** and **quantitative research** workflows. In institutional settings, portfolio managers and research analysts face the challenge of synthesizing signals from multiple data sources — price history, risk metrics, forward-looking models, and qualitative analyst views — into coherent investment recommendations at scale.

The agent architecture mirrors how a real research desk operates: a data team pulls and normalizes market data, a quant team produces factor exposures and risk metrics, a systematic strategy generates model signals, an analyst overlay adds qualitative judgment, a risk team flags concentration or volatility concerns, and a senior PM synthesizes everything into a briefing for the investment committee.

By automating this workflow with LangGraph and LLMs, the system demonstrates how **AI agent orchestration** can compress a multi-day research process into a repeatable, auditable, sub-minute pipeline — with clear methodology, transparent scoring, and human-readable outputs. This has practical applications in **algorithmic trading support**, **automated portfolio monitoring**, **risk reporting**, and **client-facing investment communications**.

---

## Author

**[@negrsm](https://github.com/negrsm)**

---

*Built with LangGraph · scikit-learn · OpenAI · Streamlit*
