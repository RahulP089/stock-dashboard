import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page Configuration
st.set_page_config(page_title="Advanced Indian Stock Dashboard", page_icon="⚡", layout="wide")

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

# 1. Market Direction Banner
st.markdown("### 📈 Overall Market Direction")
m1, m2, m3, m4 = st.columns(4)
m1.metric("NIFTY 50", f"{n_close:,.2f}", f"{n_pct:.2f}%")
m2.metric("BANK NIFTY", f"{b_close:,.2f}", f"{b_pct:.2f}%")
m3.info("Market Breadth Check: Ensure overall trend supports your entry.")
st.markdown("---")

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
st.sidebar.markdown("---")

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

    # Calculate IDEAL historical RSI for this specific stock
    # We use the 15th percentile of RSI as the historical "bounce" zone
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

    # Support & Resistance Metrics
    high_52w = full_data["High"].max()
    full_data["Daily_Range_Pct"] = ((full_data["High"] - full_data["Low"]) / full_data["Close"]) * 100
    avg_1day_volatility_pct = float(full_data["Daily_Range_Pct"].tail(20).mean())

    # Data Slice for Selected Horizon
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

    # Volume Logic
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

    # RSI Logic
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

    # Trend Logic (EMA 50 & 200)
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

    # MACD Score
    if macd > signal_line and macd > 0:
        macd_score = 15
        trade_reasons.append("✅ **MACD:** Bullish crossover and trending above the zero line.")
    elif macd > signal_line:
        macd_score = 5
        trade_reasons.append("✅ **MACD:** Triggered a bullish crossover, but still recovering below zero.")
    else:
        macd_score = -10
        trade_reasons.append("❌ **MACD:** Bearish momentum (trading below the signal line).")
    
    # VWAP Score
    if current_price > current_vwap:
        vwap_score = 10
        trade_reasons.append("✅ **Intraday Bias:** Price closed above the VWAP, indicating strong intraday buying pressure.")
    else:
        vwap_score = -5
        trade_reasons.append("❌ **Intraday Bias:** Price closed below VWAP, indicating intraday sellers pushed it down.")

    # Final Score Calculation
    raw_score = trend_score + rsi_score + vol_score + macd_score + vwap_score
    trade_score = int(max(0, min(100, raw_score)))

    # Current Day Projections
    pivot_point = (prev_high + prev_low + prev_close) / 3.0
    possible_high_pivot = (2 * pivot_point) - prev_low
    possible_low_pivot = (2 * pivot_point) - prev_high
    possible_high_atr = current_price + (1.0 * atr)
    possible_low_atr = max(0, current_price - (1.0 * atr))
    expected_day_high = (possible_high_pivot + possible_high_atr) / 2
    expected_day_low = (possible_low_pivot + possible_low_atr) / 2

    # SAFE DIP ENTRY CALCULATIONS
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
    # 4. DASHBOARD UI LAYOUT
    # ==========================
    st.markdown(f"### Stock: **{symbol}** | Horizon Target: **{selected_timeframe_label}**")

    col1, col2, col3, col4, col5 = st.columns(5)
    
    col1.metric("Current Market Price", f"₹{current_price:.2f}")
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
    
    with st.expander("📝 **Why did it get this score? (Trade Rationale Breakdown)**", expanded=True):
        st.write("Here is the detailed technical breakdown driving this recommendation based on core Indian market strategies:")
        for reason in trade_reasons:
            st.write(reason)

    st.markdown("---")

    # SUPPORT & RESISTANCE BOX
    st.subheader("🧱 Key Support & Resistance Levels")
    sr1, sr2, sr3, sr4 = st.columns(4)
    sr1.metric("Previous Day High", f"₹{prev_high:.2f}")
    sr2.metric("Previous Day Low", f"₹{prev_low:.2f}")
    sr3.metric("52-Week High", f"₹{high_52w:.2f}")
    sr4.metric("VWAP (Intraday Bias)", f"₹{current_vwap:.2f}", "Bullish" if current_price > current_vwap else "Bearish")

    st.markdown("---")

    # CURRENT DAY POSSIBLE HIGH & LOW
    st.subheader("📊 Today's Expected Price Range")
    h_col1, h_col2, h_col3, h_col4 = st.columns(4)
    h_col1.metric("Expected Day High", f"₹{expected_day_high:.2f}", f"+{((expected_day_high - current_price)/current_price)*100:.2f}%")
    h_col2.metric("Expected Day Low", f"₹{expected_day_low:.2f}", f"{((expected_day_low - current_price)/current_price)*100:.2f}%")
    h_col3.metric("Pivot Point (P)", f"₹{pivot_point:.2f}")
    h_col4.metric("14-Day ATR Range", f"±₹{atr:.2f}")

    st.markdown("---")

    # SAFE ENTRY RECOMMENDATION BOX
    st.subheader("🛡️ Safe Dip Entry & Risk-Free Price Point")
    entry_col1, entry_col2, entry_col3, entry_col4, entry_col5 = st.columns(5)
    
    entry_col1.metric("Current Price (CMP)", f"₹{current_price:.2f}")
    entry_col2.metric("SAFE DIP BUY PRICE", f"₹{safe_entry_price:.2f}", f"-{safe_dip_discount_pct:.2f}% Dip")
    entry_col3.metric(f"Target Exit (₹{target_profit} Profit)", f"₹{safe_target_price:.2f}")
    entry_col4.metric("Strict Stop Loss", f"₹{stop_loss_price:.2f}")
    
    # NEW IDEAL RSI METRIC
    entry_col5.metric("Ideal Historical Entry RSI", ideal_rsi_zone)

    st.markdown("---")

    left_col, right_col = st.columns([1.1, 0.9])
    with left_col:
        st.subheader("📊 Key Exponential Moving Averages (EMA)")
        st.caption("Institutions rely heavily on EMAs for swing and long-term trends.")
        m1, m2, m3 = st.columns(3)
        m1.metric("20 EMA (Short-Term)", f"₹{ema20:.2f}")
        m2.metric("50 EMA (Swing Trend)", f"₹{ema50:.2f}")
        m3.metric("200 EMA (Long-Term)", f"₹{ema200:.2f}")
        st.info(f"Trend Analysis: {trend_status}")

    with right_col:
        # TIMEFRAME RECOMMENDATION
        if avg_1day_volatility_pct > 0:
            est_days_needed = int(np.ceil(req_move_pct / (avg_1day_volatility_pct * 0.5)))
        else:
            est_days_needed = 10
            
        st.subheader(f"🎯 Profit Horizon Advisory")
        st.write(f"Required Surge: **+{req_move_pct:.2f}%** to hit ₹{target_profit} profit.")
        if trading_days_count >= est_days_needed:
            st.success(f"✅ Your active timeframe is well suited. Based on daily volatility ({avg_1day_volatility_pct:.2f}%/day), profit is realistic within ~{est_days_needed} days.")
        else:
            st.warning(f"⚠️ Recommendation: Increase horizon. Reaching your target requires ~{est_days_needed} days. Current timeframe may be too short.")

    st.markdown("---")

    # ==========================================
    # SELECTED TIMEFRAME HISTORICAL ANALYSIS
    # ==========================================
    st.subheader(f"📅 Historical Performance: {selected_timeframe_label}")
    
    period_high = data_horizon["High"].max()
    period_low = data_horizon["Low"].min()
    period_mean = data_horizon["Close"].mean()
    period_median = data_horizon["Close"].median()
    start_price = float(data_horizon["Close"].iloc[0])
    end_price = float(data_horizon["Close"].iloc[-1])
    period_return_pct = ((end_price - start_price) / start_price) * 100
    avg_period_volume = data_horizon["Volume"].mean()
    
    t_col1, t_col2, t_col3, t_col4, t_col5, t_col6 = st.columns(6)
    t_col1.metric("Period High", f"₹{period_high:.2f}")
    t_col2.metric("Period Low", f"₹{period_low:.2f}")
    t_col3.metric("Average Price", f"₹{period_mean:.2f}")
    t_col4.metric("Median Price", f"₹{period_median:.2f}")
    t_col5.metric("Period Return", f"₹{end_price:.2f}", f"{period_return_pct:.2f}% over {len(data_horizon)} days")
    t_col6.metric("Avg Daily Vol", f"{int(avg_period_volume):,}")

    st.markdown("---")

    # ==========================================
    # PROFIT MINING EXPECTED DAYS MATRIX
    # ==========================================
    st.subheader("⛏️ Profit Mining Matrix")
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

    mining_df = pd.DataFrame(mining_data)
    st.dataframe(mining_df, use_container_width=True, hide_index=True)
    st.markdown("---")

    # ==========================================
    # CHARTING
    # ==========================================
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
    
    # EMAs and VWAP plotted
    fig.add_trace(go.Scatter(x=data_horizon.index, y=data_horizon["EMA20"], mode="lines", name="EMA 20", line=dict(color='blue', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=data_horizon.index, y=data_horizon["EMA50"], mode="lines", name="EMA 50", line=dict(color='orange', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=data_horizon.index, y=data_horizon["EMA200"], mode="lines", name="EMA 200", line=dict(color='red', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=data_horizon.index, y=data_horizon["VWAP"], mode="lines", name="VWAP", line=dict(color='yellow', width=2, dash='dot')), row=1, col=1)

    fig.add_hline(y=expected_day_high, line_dash="dot", line_color="lime", row=1, col=1, 
                  annotation_text=f"Exp High (₹{expected_day_high:.2f})")
    fig.add_hline(y=expected_day_low, line_dash="dot", line_color="crimson", row=1, col=1, 
                  annotation_text=f"Exp Low (₹{expected_day_low:.2f})")

    fig.add_trace(go.Scatter(
        x=data_horizon.index, y=data_horizon["RSI"], mode="lines", name="RSI", line=dict(color='purple')
    ), row=2, col=1)

    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1, annotation_text="Overbought (70)")
    fig.add_hline(y=60, line_dash="dash", line_color="orange", row=2, col=1, annotation_text="Bullish (60)")
    fig.add_hline(y=40, line_dash="dash", line_color="yellow", row=2, col=1, annotation_text="Weak (40)")
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1, annotation_text="Oversold (30)")

    fig.update_layout(
        template="plotly_dark", 
        height=750, 
        xaxis_rangeslider_visible=False,
        xaxis2_rangeslider_visible=False,
        showlegend=True,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
