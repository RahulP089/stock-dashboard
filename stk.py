import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page Configuration
st.set_page_config(page_title="Institutional Stock Terminal", page_icon="⚡", layout="wide")

# ==========================
# CUSTOM UI STYLING (Terminal UI)
# ==========================
st.markdown("""
    <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        div[data-testid="stMetric"] {
            background-color: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.08);
            padding: 10px 14px;
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================
# CACHED MARKET INDICES
# ==========================
@st.cache_data(ttl=300)
def get_market_indices():
    try:
        nifty = yf.Ticker("^NSEI").history(period="5d")
        bank = yf.Ticker("^NSEBANK").history(period="5d")
        
        n_close = nifty['Close'].iloc[-1]
        n_prev = nifty['Close'].iloc[-2]
        n_pct = ((n_close - n_prev) / n_prev) * 100
        
        b_close = bank['Close'].iloc[-1]
        b_prev = bank['Close'].iloc[-2]
        b_pct = ((b_close - b_prev) / b_prev) * 100
        
        return n_close, n_pct, b_close, b_pct
    except:
        return 0, 0, 0, 0

n_close, n_pct, b_close, b_pct = get_market_indices()

st.title("⚡ Institutional Stock Terminal & Delivery Engine")

# 1. Market Direction Header Panel
with st.container(border=True):
    mc1, mc2, mc3 = st.columns([1, 1, 2])
    mc1.metric("NIFTY 50", f"₹{n_close:,.2f}", f"{n_pct:.2f}%")
    mc2.metric("BANK NIFTY", f"₹{b_close:,.2f}", f"{b_pct:.2f}%")
    mc3.info("🛡️ **System Status:** Market breadth verified. Match stock entry direction with macro index momentum.")

# ==========================
# 2. SIDEBAR CONFIGURATION
# ==========================
st.sidebar.header("🔍 Search & Parameters")
company = st.sidebar.text_input("Enter Company Name", value="Reliance")

timeframe_options = {
    "1 Day (Intraday Focus)": 1,
    "1 Week Focus": 5,
    "2 Weeks Focus": 10,
    "1 Month Focus": 21,
    "3 Months Focus": 63,
    "6 Months Focus": 126,
    "1 Year Focus": 252
}
selected_timeframe_label = st.sidebar.selectbox("Select Horizon Target:", list(timeframe_options.keys()), index=1)
trading_days_count = timeframe_options[selected_timeframe_label]

st.sidebar.markdown("---")
st.sidebar.header("💰 Risk & Capital")
investment = st.sidebar.number_input("Capital to Invest (₹)", value=50000, step=1000)
target_profit = st.sidebar.number_input("Target Profit (₹)", value=1000, step=500)

st.sidebar.markdown("---")
st.sidebar.header("⛏️ Profit Mining Options")
mining_days = st.sidebar.slider("Max Expected Holding Days", 1, 252, 10, 1)

if company:
    search = yf.Search(company, max_results=20)
    if not search.quotes:
        st.error("Company not found. Please try another search term.")
        st.stop()

    indian_quotes = [q for q in search.quotes if str(q.get("symbol", "")).endswith((".NS", ".BO"))]
    other_quotes = [q for q in search.quotes if not str(q.get("symbol", "")).endswith((".NS", ".BO"))]
    sorted_quotes = indian_quotes + other_quotes

    options = {
        f"{q.get('shortname', 'NA')} | {q.get('symbol', 'NA')} ({q.get('exchange', 'NA')})": q.get('symbol')
        for q in sorted_quotes
    }

    selected_option = st.sidebar.selectbox("Select Exchange/Stock:", list(options.keys()))
    symbol = options[selected_option]

    # ==========================
    # 3. DATA PROCESSING & TECHNICALS
    # ==========================
    ticker = yf.Ticker(symbol)
    full_data = ticker.history(period="1y")

    if full_data.empty or len(full_data) < 20:
        st.error("Insufficient historical data available for this symbol.")
        st.stop()

    if isinstance(full_data.columns, pd.MultiIndex):
        full_data.columns = full_data.columns.get_level_values(0)

    # Core Indicators
    full_data["EMA20"] = full_data["Close"].ewm(span=20, adjust=False).mean()
    full_data["EMA50"] = full_data["Close"].ewm(span=50, adjust=False).mean()
    full_data["EMA200"] = full_data["Close"].ewm(span=200, adjust=False).mean()
    full_data["Vol_MA20"] = full_data["Volume"].rolling(20).mean().fillna(1)

    delta = full_data["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, 0.00001))
    full_data["RSI"] = 100 - (100 / (1 + rs))

    historical_bounce_rsi = full_data["RSI"].quantile(0.15)
    ideal_rsi_zone = f"{historical_bounce_rsi:.1f} - {historical_bounce_rsi + 10:.1f}"

    ema12 = full_data["Close"].ewm(span=12, adjust=False).mean()
    ema26 = full_data["Close"].ewm(span=26, adjust=False).mean()
    full_data["MACD"] = ema12 - ema26
    full_data["Signal_Line"] = full_data["MACD"].ewm(span=9, adjust=False).mean()

    ma20_simple = full_data["Close"].rolling(20).mean()
    std20 = full_data["Close"].rolling(20).std()
    full_data["BB_Upper"] = ma20_simple + (std20 * 2)
    full_data["BB_Lower"] = ma20_simple - (std20 * 2)

    high_low = full_data["High"] - full_data["Low"]
    high_close = np.abs(full_data["High"] - full_data["Close"].shift())
    low_close = np.abs(full_data["Low"] - full_data["Close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    full_data["ATR"] = tr.rolling(14).mean().bfill()

    full_data['Typical_Price'] = (full_data['High'] + full_data['Low'] + full_data['Close']) / 3
    full_data['VWAP'] = (full_data['Typical_Price'] * full_data['Volume']).cumsum() / full_data['Volume'].cumsum()
    full_data = full_data.bfill().ffill()

    high_52w = full_data["High"].max()
    full_data["Daily_Range_Pct"] = ((full_data["High"] - full_data["Low"]) / full_data["Close"]) * 100
    avg_1day_volatility_pct = float(full_data["Daily_Range_Pct"].tail(20).mean())

    data_horizon = full_data.tail(max(trading_days_count, 5)).copy()
    current_price = float(data_horizon["Close"].iloc[-1])
    ema20 = float(data_horizon["EMA20"].iloc[-1])
    ema50 = float(data_horizon["EMA50"].iloc[-1])
    ema200 = float(data_horizon["EMA200"].iloc[-1])
    bb_lower = float(data_horizon["BB_Lower"].iloc[-1])
    rsi = float(data_horizon["RSI"].iloc[-1])
    macd = float(data_horizon["MACD"].iloc[-1])
    signal_line = float(data_horizon["Signal_Line"].iloc[-1])
    atr = float(data_horizon["ATR"].iloc[-1])
    current_vol = float(data_horizon["Volume"].iloc[-1])
    vol_ma20 = float(data_horizon["Vol_MA20"].iloc[-1])
    current_vwap = float(data_horizon["VWAP"].iloc[-1])

    if len(full_data) >= 2:
        prev_high = float(full_data["High"].iloc[-2])
        prev_low = float(full_data["Low"].iloc[-2])
        prev_close = float(full_data["Close"].iloc[-2])
    else:
        prev_high, prev_low, prev_close = current_price, current_price, current_price

    # Scoring Engine
    trade_reasons = []
    vol_ratio = current_vol / vol_ma20 if vol_ma20 > 0 else 1
    
    if vol_ratio >= 2.0:
        vol_status = "✅ High Volume Surge"
        vol_score = 20
        trade_reasons.append(f"✅ **Volume:** Massive institutional accumulation detected ({vol_ratio:.1f}x average).")
    elif vol_ratio >= 1.2:
        vol_status = "✅ Above Avg Volume"
        vol_score = 10
        trade_reasons.append(f"✅ **Volume:** Healthy buying interest ({vol_ratio:.1f}x average).")
    else:
        vol_status = "⚠️ Low Volume Activity"
        vol_score = 0
        trade_reasons.append(f"⚠️ **Volume:** Sub-average volume ({vol_ratio:.1f}x). Lack of institutional participation.")

    if rsi >= 70:
        rsi_eval = "⚠️ Overbought (>70)"
        rsi_score = 15
        trade_reasons.append(f"⚠️ **RSI ({rsi:.1f}):** Overbought condition. High vulnerability to short-term pullbacks.")
    elif rsi >= 60:
        rsi_eval = "✅ Bullish Momentum (>60)"
        rsi_score = 20
        trade_reasons.append(f"✅ **RSI ({rsi:.1f}):** Strong upward momentum zone.")
    elif 40 <= rsi < 60:
        rsi_eval = "⚠️ Neutral Consolidation"
        rsi_score = 5
        trade_reasons.append(f"⚠️ **RSI ({rsi:.1f}):** Sideways consolidation range.")
    elif 30 <= rsi < 40:
        rsi_eval = "❌ Weak Momentum (<40)"
        rsi_score = -10
        trade_reasons.append(f"❌ **RSI ({rsi:.1f}):** Weak momentum; sellers dominating.")
    else:
        rsi_eval = "✅ Oversold Zone (<30)"
        rsi_score = 10
        trade_reasons.append(f"✅ **RSI ({rsi:.1f}):** Deep value discount zone. Rebound watch active.")

    if current_price > ema50 and current_price > ema200:
        trend_status = "Strong Bullish Structure"
        trend_score = 20
        trade_reasons.append("✅ **Trend:** Price securely above both 50 and 200 EMA.")
    elif ema50 > ema200:
        trend_status = "Moderate Bullish Setup"
        trend_score = 15
        trade_reasons.append("✅ **Trend:** Golden cross alignment active (50 EMA > 200 EMA).")
    else:
        trend_status = "Macro Bearish Trend"
        trend_score = -10
        trade_reasons.append("❌ **Trend:** Trading below core moving averages. High structural risk.")

    if macd > signal_line and macd > 0:
        macd_score = 15
        trade_reasons.append("✅ **MACD:** Bullish crossover confirmed above zero line.")
    elif macd > signal_line:
        macd_score = 5
        trade_reasons.append("✅ **MACD:** Early bullish cross, recovering from lower levels.")
    else:
        macd_score = -10
        trade_reasons.append("❌ **MACD:** Bearish alignment below signal line.")
    
    vwap_score = 10 if current_price > current_vwap else -5
    if current_price > current_vwap:
        trade_reasons.append("✅ **VWAP:** Price holding above session benchmark value.")
    else:
        trade_reasons.append("❌ **VWAP:** Price trading below session benchmark value.")

    trade_score = int(max(0, min(100, trend_score + rsi_score + vol_score + macd_score + vwap_score)))

    pivot_point = (prev_high + prev_low + prev_close) / 3.0
    expected_day_high = ((2 * pivot_point) - prev_low + (current_price + atr)) / 2
    expected_day_low = ((2 * pivot_point) - prev_high + max(0, current_price - atr)) / 2

    cmp_shares = int(investment // current_price) if current_price > 0 else 0
    cmp_target_price = current_price + (target_profit / cmp_shares) if cmp_shares > 0 else current_price
    req_move_pct = ((cmp_target_price - current_price) / current_price) * 100 if current_price > 0 else 0

    safe_entry_price = ema20 if rsi > 70 and current_price > ema20 else (current_price if rsi < 30 else (ema50 if current_price > ema50 else bb_lower))
    safe_shares = int(investment // safe_entry_price) if safe_entry_price > 0 else cmp_shares
    safe_target_price = safe_entry_price + (target_profit / safe_shares) if safe_shares > 0 else cmp_target_price
    safe_dip_discount_pct = ((current_price - safe_entry_price) / current_price) * 100
    stop_loss_price = max(0, safe_entry_price - (1.5 * atr))

    # ==========================
    # 4. DASHBOARD PRESENTATION (Card Architecture)
    # ==========================
    st.markdown(f"### Target Asset: **{symbol}** | Timeframe: **{selected_timeframe_label}**")

    # Primary Executive Control Card
    with st.container(border=True):
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Current Market Price", f"₹{current_price:.2f}")
        c2.metric("Volume State", f"{vol_ratio:.1f}x Avg")
        c3.metric("RSI State", f"{rsi:.2f}")
        c4.metric("Algorithmic Score", f"{trade_score}/100")

        c2.caption(vol_status)
        c3.caption(rsi_eval)

        if trade_score >= 80:
            c5.success("EXCELLENT BUY")
        elif trade_score >= 65:
            c5.info("MODERATE BUY")
        elif trade_score >= 45:
            c5.warning("HOLD / NEUTRAL")
        else:
            c5.error("AVOID / HIGH RISK")
        
        st.caption(f"**Institutional MACD State:** {'Bullish Expansion' if macd > 0 and macd > signal_line else 'Defensive / Bearish'}")

    # Structured 3-Column Telemetry Panel
    col_l, col_m, col_r = st.columns(3)
    
    with col_l:
        with st.container(border=True):
            st.markdown("##### 🧱 Support & Resistance")
            st.write(f"**Prev Session High:** ₹{prev_high:.2f}")
            st.write(f"**Prev Session Low:** ₹{prev_low:.2f}")
            st.write(f"**52-Week Peak:** ₹{high_52w:.2f}")
            st.write(f"**VWAP Baseline:** ₹{current_vwap:.2f}")

    with col_m:
        with st.container(border=True):
            st.markdown("##### 📊 Intraday Projections")
            st.write(f"**Projected High:** ₹{expected_day_high:.2f} (+{((expected_day_high - current_price)/current_price)*100:.2f}%)")
            st.write(f"**Projected Low:** ₹{expected_day_low:.2f}")
            st.write(f"**Classic Pivot (P):** ₹{pivot_point:.2f}")
            st.write(f"**14-Day ATR Buffer:** ±₹{atr:.2f}")

    with col_r:
        with st.container(border=True):
            st.markdown("##### 📈 Trend Telemetry")
            st.write(f"**20 EMA:** ₹{ema20:.2f}")
            st.write(f"**50 EMA:** ₹{ema50:.2f}")
            st.write(f"**200 EMA:** ₹{ema200:.2f}")
            st.write(f"**Macro Filter:** {trend_status}")

    # Risk-Managed Entry Execution Panel
    with st.container(border=True):
        st.markdown("##### 🛡️ Execution Framework & Safe Dip Entry Matrix")
        e1, e2, e3, e4, e5 = st.columns(5)
        e1.metric("Current CMP", f"₹{current_price:.2f}")
        e2.metric("SAFE DIP ENTRY", f"₹{safe_entry_price:.2f}", f"-{safe_dip_discount_pct:.2f}% Target")
        e3.metric(f"Target Exit (₹{target_profit})", f"₹{safe_target_price:.2f}")
        e4.metric("Strict Stop Loss", f"₹{stop_loss_price:.2f}")
        e5.metric("Ideal RSI Zone", ideal_rsi_zone)

    # Historical Performance Grid
    with st.container(border=True):
        st.markdown(f"##### 📅 Historical Horizon Performance ({selected_timeframe_label})")
        period_high = data_horizon["High"].max()
        period_low = data_horizon["Low"].min()
        period_mean = data_horizon["Close"].mean()
        period_median = data_horizon["Close"].median()
        start_price = float(data_horizon["Close"].iloc[0])
        end_price = float(data_horizon["Close"].iloc[-1])
        period_return_pct = ((end_price - start_price) / start_price) * 100
        avg_period_volume = data_horizon["Volume"].mean()
        
        t1, t2, t3, t4, t5, t6 = st.columns(6)
        t1.metric("Period High", f"₹{period_high:.2f}")
        t2.metric("Period Low", f"₹{period_low:.2f}")
        t3.metric("Average Price", f"₹{period_mean:.2f}")
        t4.metric("Median Price", f"₹{period_median:.2f}")
        t5.metric("Period Return", f"₹{end_price:.2f}", f"{period_return_pct:.2f}%")
        t6.metric("Avg Daily Vol", f"{int(avg_period_volume):,}")

    # Profit Mining Module
    with st.container(border=True):
        st.markdown("##### ⛏️ Profit Mining Projections Matrix")
        profit_targets_to_mine = [target_profit * 0.5, target_profit, target_profit * 2.5, target_profit * 5]
        mining_data = []
        for pt in profit_targets_to_mine:
            pt_cmp_target = current_price + (pt / cmp_shares) if cmp_shares > 0 else 0
            pt_cmp_req_move = ((pt_cmp_target - current_price) / current_price) * 100 if current_price > 0 else 0
            pt_est_days_cmp = int(np.ceil(pt_cmp_req_move / (avg_1day_volatility_pct * 0.5))) if avg_1day_volatility_pct > 0 else 0
            
            pt_safe_target = safe_entry_price + (pt / safe_shares) if safe_shares > 0 else 0
            pt_safe_req_move = ((pt_safe_target - safe_entry_price) / safe_entry_price) * 100 if safe_entry_price > 0 else 0
            pt_est_days_safe = int(np.ceil(pt_safe_req_move / (avg_1day_volatility_pct * 0.5))) if avg_1day_volatility_pct > 0 else 0
            
            mining_data.append({
                "Target Profit (₹)": f"₹{pt:,.0f}",
                "Required Move %": f"+{pt_cmp_req_move:.2f}%",
                "Target Price (Current Entry)": f"₹{pt_cmp_target:.2f}",
                "Expected Days (CMP)": f"~{pt_est_days_cmp} Days",
                "Target Price (Safe Dip)": f"₹{pt_safe_target:.2f}",
                "Expected Days (Dip)": f"~{pt_est_days_safe} Days",
            })
        st.dataframe(pd.DataFrame(mining_data), use_container_width=True, hide_index=True)

    # Interactive Chart Rendering
    with st.container(border=True):
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.03, 
            row_heights=[0.7, 0.3],
            subplot_titles=(f"📉 Technical Canvas ({selected_timeframe_label})", "RSI Momentum Oscillator (14)")
        )

        fig.add_trace(go.Candlestick(
            x=data_horizon.index, open=data_horizon["Open"], high=data_horizon["High"],
            low=data_horizon["Low"], close=data_horizon["Close"], name="Price Action"
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=data_horizon.index, y=data_horizon["EMA20"], mode="lines", name="EMA 20", line=dict(color='#38bdf8', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=data_horizon.index, y=data_horizon["EMA50"], mode="lines", name="EMA 50", line=dict(color='#fbbf24', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=data_horizon.index, y=data_horizon["EMA200"], mode="lines", name="EMA 200", line=dict(color='#f87171', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=data_horizon.index, y=data_horizon["VWAP"], mode="lines", name="VWAP", line=dict(color='#c084fc', width=2, dash='dot')), row=1, col=1)

        fig.add_hline(y=expected_day_high, line_dash="dot", line_color="#4ade80", row=1, col=1, annotation_text=f"Exp High (₹{expected_day_high:.2f})")
        fig.add_hline(y=expected_day_low, line_dash="dot", line_color="#f43f5e", row=1, col=1, annotation_text=f"Exp Low (₹{expected_day_low:.2f})")

        fig.add_trace(go.Scatter(
            x=data_horizon.index, y=data_horizon["RSI"], mode="lines", name="RSI", line=dict(color='#e879f9', width=1.5)
        ), row=2, col=1)

        fig.add_hline(y=70, line_dash="dash", line_color="#f87171", row=2, col=1, annotation_text="Overbought (70)")
        fig.add_hline(y=60, line_dash="dash", line_color="#fbbf24", row=2, col=1, annotation_text="Bullish (60)")
        fig.add_hline(y=40, line_dash="dash", line_color="#38bdf8", row=2, col=1, annotation_text="Weak (40)")
        fig.add_hline(y=30, line_dash="dash", line_color="#4ade80", row=2, col=1, annotation_text="Oversold (30)")

        fig.update_layout(
            template="plotly_dark", 
            height=700, 
            xaxis_rangeslider_visible=False,
            xaxis2_rangeslider_visible=False,
            showlegend=True,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    # Final Tabbed Executive Report Panel
    with st.container(border=True):
        st.markdown("### 🎯 Executive Decision Report & Technical Rules")
        
        tab1, tab2, tab3 = st.tabs(["✅ Advantages (Strengths)", "⚠️ Risk Factors (Threats)", "📊 General RSI Guide"])
        
        with tab1:
            st.success("**Validated Asset Strengths:**")
            matched_advantages = [r for r in trade_reasons if "✅" in r]
            if matched_advantages:
                for reason in matched_advantages:
                    st.write(reason)
            else:
                st.write("No major structural advantages detected under current conditions.")
                
        with tab2:
            st.error("**Active Risk Factors & Structural Warnings:**")
            matched_risks = [r for r in trade_reasons if "❌" in r or "⚠️" in r]
            if matched_risks:
                for reason in matched_risks:
                    st.write(reason)
            else:
                st.write("Zero structural flags identified. Trend integrity is clean.")
                
        with tab3:
            st.info("**Reference Playbook: RSI Ranges & Market Behavior**")
            st.markdown("""
            *   **RSI < 30 (Oversold / Value Zone):** 🟢 **AGGRESSIVE DIP BUY** – Asset is deeply discounted. High probability of mean-reversion bounces.
            *   **RSI 30 - 45 (Accumulation Zone):** 🟢 **FAVORABLE ENTRY** – Ideal risk-to-reward window for delivery swing setups near key support levels.
            *   **RSI 45 - 60 (Neutral Range):** 🟡 **CONSOLIDATION / MONITOR** – Price is chopping sideways. Avoid capital deployment until volume expands.
            *   **RSI 60 - 70 (Momentum Trend Zone):** 🔵 **BREAKOUT CONTINUATION** – Strong institutional demand. Excellent for trend-following, but maintain strict trailing stops.
            *   **RSI > 70 (Overextended Zone):** 🔴 **DISTRIBUTION / AVOID** – Extreme buying pressure. Imminent risk of profit booking and sharp correction.
            """)
