import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page Configuration
st.set_page_config(page_title="Advanced Indian Stock Dashboard", page_icon="⚡", layout="wide")

# ==========================
# CUSTOM UI STYLING (Card & Spacing Polish)
# ==========================
st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        div[data-testid="stMetric"] {
            background-color: rgba(28, 32, 44, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.08);
            padding: 12px 15px;
            border-radius: 10px;
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

st.title("⚡ Advanced Stock Delivery & Safe Dip Entry Dashboard")

# 1. Market Direction Banner (Card Style)
with st.container(border=True):
    st.markdown("##### 📈 Overall Market Direction & Breadth")
    m1, m2, m3 = st.columns([1, 1, 2])
    m1.metric("NIFTY 50", f"₹{n_close:,.2f}", f"{n_pct:.2f}%")
    m2.metric("BANK NIFTY", f"₹{b_close:,.2f}", f"{b_pct:.2f}%")
    m3.info("💡 **Macro Rule:** Always check that the overall market trend supports your stock entry before deploying capital.")

# ==========================
# 2. SIDEBAR & SEARCH
# ==========================
st.sidebar.header("🔍 Stock Search & Settings")
company = st.sidebar.text_input("Enter Company Name", value="Reliance")

timeframe_options = {
    "1 Day (Tomorrow Focus)": 1,
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
st.sidebar.header("💰 Investment Settings")
investment = st.sidebar.number_input("Capital to Invest (₹)", value=42000, step=1000)
target_profit = st.sidebar.number_input("Target Profit (₹)", value=1000, step=500)

st.sidebar.markdown("---")
st.sidebar.header("⛏️ Profit Mining Options")
mining_days = st.sidebar.slider(
    "Max Expected Holding Days", 
    min_value=1, max_value=252, value=10, step=1, 
    help="Select how many days you want to hold to see the realistic expected profit."
)

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
    st.sidebar.success(f"Active Symbol: {symbol}")

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

    # Core Moving Averages
    full_data["EMA20"] = full_data["Close"].ewm(span=20, adjust=False).mean()
    full_data["EMA50"] = full_data["Close"].ewm(span=50, adjust=False).mean()
    full_data["EMA200"] = full_data["Close"].ewm(span=200, adjust=False).mean()
    full_data["Vol_MA20"] = full_data["Volume"].rolling(20).mean().fillna(1)

    # RSI
    delta = full_data["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, 0.00001))
    full_data["RSI"] = 100 - (100 / (1 + rs))

    historical_bounce_rsi = full_data["RSI"].quantile(0.15)
    ideal_rsi_zone = f"{historical_bounce_rsi:.1f} - {historical_bounce_rsi + 10:.1f}"

    # MACD
    ema12 = full_data["Close"].ewm(span=12, adjust=False).mean()
    ema26 = full_data["Close"].ewm(span=26, adjust=False).mean()
    full_data["MACD"] = ema12 - ema26
    full_data["Signal_Line"] = full_data["MACD"].ewm(span=9, adjust=False).mean()

    # Bollinger Bands & ATR
    ma20_simple = full_data["Close"].rolling(20).mean()
    std20 = full_data["Close"].rolling(20).std()
    full_data["BB_Upper"] = ma20_simple + (std20 * 2)
    full_data["BB_Lower"] = ma20_simple - (std20 * 2)

    high_low = full_data["High"] - full_data["Low"]
    high_close = np.abs(full_data["High"] - full_data["Close"].shift())
    low_close = np.abs(full_data["Low"] - full_data["Close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    full_data["ATR"] = tr.rolling(14).mean().bfill()

    # VWAP
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

    # ==========================
    # TRADE SCORING & REASONS LOGIC
    # ==========================
    trade_reasons = []

    vol_ratio = current_vol / vol_ma20 if vol_ma20 > 0 else 1
    if vol_ratio >= 2.0:
        vol_status = "✅ HIGH VOLUME (Breakout/Strong Signal)"
        vol_score = 20
        trade_reasons.append(f"✅ **Volume:** Traded {vol_ratio:.1f}x the 20-day average. Institutional participation is highly likely.")
    elif vol_ratio >= 1.2:
        vol_status = "✅ Above Average Volume"
        vol_score = 10
        trade_reasons.append(f"✅ **Volume:** Above average ({vol_ratio:.1f}x), showing decent accumulation.")
    else:
        vol_status = "⚠️ Low Volume (Cautious Breakout)"
        vol_score = 0
        trade_reasons.append(f"⚠️ **Volume:** Below average ({vol_ratio:.1f}x). Price movements might lack conviction and fail to hold.")

    if rsi >= 70:
        rsi_eval = "⚠️ Strong Trend (>70). Wait for pullbacks."
        rsi_score = 15
        trade_reasons.append(f"⚠️ **RSI ({rsi:.1f}):** Overbought (>70). The trend is very strong, but prone to a sudden short-term pullback.")
    elif rsi >= 60:
        rsi_eval = "✅ Bullish Momentum (>60)."
        rsi_score = 20
        trade_reasons.append(f"✅ **RSI ({rsi:.1f}):** Bullish momentum zone (>60). Stock is actively trending upward.")
    elif 40 <= rsi < 60:
        rsi_eval = "⚠️ Neutral Zone (40-60). Consolidating."
        rsi_score = 5
        trade_reasons.append(f"⚠️ **RSI ({rsi:.1f}):** Neutral zone. Stock is currently consolidating or moving sideways.")
    elif 30 <= rsi < 40:
        rsi_eval = "❌ Weak Momentum (<40)."
        rsi_score = -10
        trade_reasons.append(f"❌ **RSI ({rsi:.1f}):** Weak momentum (<40). Sellers are currently in control.")
    else:
        rsi_eval = "✅ Oversold (<30). Potential bounce."
        rsi_score = 10
        trade_reasons.append(f"✅ **RSI ({rsi:.1f}):** Oversold (<30). Stock is heavily discounted, offering a potential rapid value bounce.")

    if current_price > ema50 and current_price > ema200:
        trend_status = "Strong Bullish (Price > 50 & 200 EMA)"
        trend_score = 20
        trade_reasons.append("✅ **Trend:** Price is trading above both the 50 EMA and 200 EMA, confirming a long-term bullish uptrend.")
    elif ema50 > ema200:
        trend_status = "Moderate Bullish (50 EMA > 200 EMA)"
        trend_score = 15
        trade_reasons.append("✅ **Trend:** The 50 EMA is above the 200 EMA (Golden cross structure intact).")
    else:
        trend_status = "Bearish Trend"
        trend_score = -10
        trade_reasons.append("❌ **Trend:** Price is below major EMAs, indicating a macro bearish trend.")

    if macd > signal_line and macd > 0:
        macd_score = 15
        trade_reasons.append("✅ **MACD:** Bullish crossover and trending above the zero line.")
    elif macd > signal_line:
        macd_score = 5
        trade_reasons.append("✅ **MACD:** Triggered a bullish crossover, but still recovering below zero.")
    else:
        macd_score = -10
        trade_reasons.append("❌ **MACD:** Bearish momentum (trading below the signal line).")
    
    if current_price > current_vwap:
        vwap_score = 10
        trade_reasons.append("✅ **Intraday Bias:** Price closed above the VWAP, indicating strong intraday buying pressure.")
    else:
        vwap_score = -5
        trade_reasons.append("❌ **Intraday Bias:** Price closed below VWAP, indicating intraday sellers pushed it down.")

    raw_score = trend_score + rsi_score + vol_score + macd_score + vwap_score
    trade_score = int(max(0, min(100, raw_score)))

    pivot_point = (prev_high + prev_low + prev_close) / 3.0
    possible_high_pivot = (2 * pivot_point) - prev_low
    possible_low_pivot = (2 * pivot_point) - prev_high
    possible_high_atr = current_price + (1.0 * atr)
    possible_low_atr = max(0, current_price - (1.0 * atr))
    expected_day_high = (possible_high_pivot + possible_high_atr) / 2
    expected_day_low = (possible_low_pivot + possible_low_atr) / 2

    cmp_shares = int(investment // current_price) if current_price > 0 else 0
    cmp_target_price = current_price + (target_profit / cmp_shares) if cmp_shares > 0 else current_price
    req_move_pct = ((cmp_target_price - current_price) / current_price) * 100 if current_price > 0 else 0

    if rsi > 70:
        safe_entry_price = ema20 if current_price > ema20 else ema50
    elif rsi < 30:
        safe_entry_price = current_price
    else:
        safe_entry_price = ema50 if current_price > ema50 else bb_lower

    safe_shares = int(investment // safe_entry_price) if safe_entry_price > 0 else cmp_shares
    safe_target_price = safe_entry_price + (target_profit / safe_shares) if safe_shares > 0 else cmp_target_price
    safe_dip_discount_pct = ((current_price - safe_entry_price) / current_price) * 100
    stop_loss_price = max(0, safe_entry_price - (1.5 * atr))

    # ==========================
    # 4. DASHBOARD UI LAYOUT (Card Structure)
    # ==========================
    st.markdown(f"### Stock Analysis: **{symbol}** (`{selected_timeframe_label}`)")

    # Main Metrics Card
    with st.container(border=True):
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Current Price", f"₹{current_price:.2f}")
        col2.metric("Volume Analysis", f"{vol_ratio:.1f}x Avg")
        col3.metric("RSI (14)", f"{rsi:.2f}")
        col4.metric("Trade Score", f"{trade_score}/100")

        col2.caption(vol_status)
        col3.caption(rsi_eval)

        if trade_score >= 80:
            col5.success("EXCELLENT BUY")
        elif trade_score >= 65:
            col5.info("MODERATE BUY")
        elif trade_score >= 45:
            col5.warning("HOLD / NEUTRAL")
        else:
            col5.error("AVOID / HIGH RISK")
        
        st.caption(f"**MACD Status:** {'Bullish (Above 0)' if macd > 0 and macd > signal_line else 'Bearish or Weak'}")

    # Breakdown Expandable Container
    with st.expander("📝 **Technical Scoring Rationale & Breakdown**", expanded=False):
        for reason in trade_reasons:
            st.write(reason)

    # 3-Column Structured Layout for Secondary Metrics
    col_l, col_m, col_r = st.columns(3)
    
    with col_l:
        with st.container(border=True):
            st.markdown("##### 🧱 Support & Resistance")
            st.write(f"**Prev High:** ₹{prev_high:.2f}")
            st.write(f"**Prev Low:** ₹{prev_low:.2f}")
            st.write(f"**52-W High:** ₹{high_52w:.2f}")
            st.write(f"**VWAP Bias:** ₹{current_vwap:.2f} ({'Bullish' if current_price > current_vwap else 'Bearish'})")

    with col_m:
        with st.container(border=True):
            st.markdown("##### 📊 Today's Expected Range")
            st.write(f"**Exp High:** ₹{expected_day_high:.2f} (+{((expected_day_high - current_price)/current_price)*100:.2f}%)")
            st.write(f"**Exp Low:** ₹{expected_day_low:.2f} ({((expected_day_low - current_price)/current_price)*100:.2f}%)")
            st.write(f"**Pivot Point:** ₹{pivot_point:.2f}")
            st.write(f"**ATR Range:** ±₹{atr:.2f}")

    with col_r:
        with st.container(border=True):
            st.markdown("##### 📈 Trend & Horizon")
            st.write(f"**20 EMA:** ₹{ema20:.2f}")
            st.write(f"**50 EMA:** ₹{ema50:.2f}")
            st.write(f"**200 EMA:** ₹{ema200:.2f}")
            st.write(f"**Macro Status:** {trend_status}")

    # Safe Dip Entry Card
    with st.container(border=True):
        st.markdown("##### 🛡️ Safe Dip Entry & Risk-Free Price Point")
        e1, e2, e3, e4, e5 = st.columns(5)
        e1.metric("Current Price (CMP)", f"₹{current_price:.2f}")
        e2.metric("SAFE DIP BUY PRICE", f"₹{safe_entry_price:.2f}", f"-{safe_dip_discount_pct:.2f}% Dip")
        e3.metric(f"Target Exit (₹{target_profit} Profit)", f"₹{safe_target_price:.2f}")
        e4.metric("Strict Stop Loss", f"₹{stop_loss_price:.2f}")
        e5.metric("Ideal Historical RSI", ideal_rsi_zone)

    # Historical Performance Card
    with st.container(border=True):
        st.markdown(f"##### 📅 Historical Performance ({selected_timeframe_label})")
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

    # Profit Mining Matrix Container
    with st.container(border=True):
        st.markdown("##### ⛏️ Profit Mining Matrix")
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
                "Expected Days (Current)": f"~{pt_est_days_cmp} Days",
                "Target Price (Safe Dip)": f"₹{pt_safe_target:.2f}",
                "Expected Days (Safe Dip)": f"~{pt_est_days_safe} Days",
            })
        st.dataframe(pd.DataFrame(mining_data), use_container_width=True, hide_index=True)

    # Interactive Chart Container
    with st.container(border=True):
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.03, 
            row_heights=[0.7, 0.3],
            subplot_titles=(f"📉 Technical Chart ({selected_timeframe_label})", "RSI (14)")
        )

        fig.add_trace(go.Candlestick(
            x=data_horizon.index, open=data_horizon["Open"], high=data_horizon["High"],
            low=data_horizon["Low"], close=data_horizon["Close"], name="Price"
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=data_horizon.index, y=data_horizon["EMA20"], mode="lines", name="EMA 20", line=dict(color='blue', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=data_horizon.index, y=data_horizon["EMA50"], mode="lines", name="EMA 50", line=dict(color='orange', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=data_horizon.index, y=data_horizon["EMA200"], mode="lines", name="EMA 200", line=dict(color='red', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=data_horizon.index, y=data_horizon["VWAP"], mode="lines", name="VWAP", line=dict(color='yellow', width=2, dash='dot')), row=1, col=1)

        fig.add_hline(y=expected_day_high, line_dash="dot", line_color="lime", row=1, col=1, annotation_text=f"Exp High (₹{expected_day_high:.2f})")
        fig.add_hline(y=expected_day_low, line_dash="dot", line_color="crimson", row=1, col=1, annotation_text=f"Exp Low (₹{expected_day_low:.2f})")

        fig.add_trace(go.Scatter(
            x=data_horizon.index, y=data_horizon["RSI"], mode="lines", name="RSI", line=dict(color='purple')
        ), row=2, col=1)

        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1, annotation_text="Overbought (70)")
        fig.add_hline(y=60, line_dash="dash", line_color="orange", row=2, col=1, annotation_text="Bullish (60)")
        fig.add_hline(y=40, line_dash="dash", line_color="yellow", row=2, col=1, annotation_text="Weak (40)")
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1, annotation_text="Oversold (30)")

        fig.update_layout(
            template="plotly_dark", 
            height=700, 
            xaxis_rangeslider_visible=False,
            xaxis2_rangeslider_visible=False,
            showlegend=True,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    # Final Trade Summary & Risk-Reward Report Container
    with st.container(border=True):
        st.markdown("### 🎯 Final Trade Summary & Risk-Reward Report")
        
        tab1, tab2, tab3 = st.tabs(["✅ Advantages (Strengths)", "⚠️ Risk Factors (Threats)", "📊 General RSI Guide"])
        
        with tab1:
            st.success("**Key Strengths for this Stock:**")
            matched_advantages = [r for r in trade_reasons if "✅" in r]
            if matched_advantages:
                for reason in matched_advantages:
                    st.write(reason)
            else:
                st.write("No major technical advantages found at the current price level.")
                
        with tab2:
            st.error("**Major Risk Factors:**")
            matched_risks = [r for r in trade_reasons if "❌" in r or "⚠️" in r]
            if matched_risks:
                for reason in matched_risks:
                    st.write(reason)
            else:
                st.write("No major technical warnings found. Trend looks solid.")
                
        with tab3:
            st.info("**How to interpret RSI (Relative Strength Index) Zones for Delivery Trades:**")
            st.markdown("""
            *   **RSI < 30 (Oversold / Extreme Value Zone):** 🟢 **STRONG BUY / DIP ENTRY** – The stock has been heavily sold off. High probability of an immediate technical bounce.
            *   **RSI 30 - 45 (Approaching Oversold / Value Dip):** 🟢 **GOOD BUY ZONE** – Favorable risk-to-reward ratio. Often the safest zone for delivery entries near support.
            *   **RSI 45 - 60 (Neutral / Consolidation Zone):** 🟡 **HOLD / WAIT** – Sideways movement. Wait for a clear volume breakout or a dip before committing capital.
            *   **RSI 60 - 70 (Strong Bullish Momentum Zone):** 🔵 **RIDE THE TREND / ACCUMULATE** – Strong buying momentum. Good for breakout trades, but keep stop-losses tight.
            *   **RSI > 70 (Overbought / Risk Zone):** 🔴 **DO NOT BUY / TAKE PROFIT** – Extended price action. High chance of profit-booking and a sudden pullback.
            """)
