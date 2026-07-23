import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page Configuration
st.set_page_config(page_title="Advanced Stock Dashboard", page_icon="⚡", layout="wide")

st.title("⚡ Advanced Stock Delivery & Safe Dip Entry Dashboard")

# ==========================
# 1. SIDEBAR & SEARCH
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

if company:
    search = yf.Search(company, max_results=20)

    if not search.quotes:
        st.error("Company not found. Please try another search term.")
        st.stop()

    # Prioritize NSE (.NS) and BSE (.BO) symbols
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
    # 2. DATA PROCESSING & TECHNICALS
    # ==========================
    ticker = yf.Ticker(symbol)
    full_data = ticker.history(period="1y")

    if full_data.empty or len(full_data) < 20:
        st.error("Insufficient historical data available for this symbol.")
        st.stop()

    if isinstance(full_data.columns, pd.MultiIndex):
        full_data.columns = full_data.columns.get_level_values(0)

    # Technical Indicators
    full_data["MA20"] = full_data["Close"].rolling(20).mean()
    full_data["MA50"] = full_data["Close"].rolling(50).mean()
    full_data["MA100"] = full_data["Close"].rolling(100).mean()
    full_data["Vol_MA20"] = full_data["Volume"].rolling(20).mean().fillna(1)

    # RSI
    delta = full_data["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, 0.00001))
    full_data["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = full_data["Close"].ewm(span=12, adjust=False).mean()
    ema26 = full_data["Close"].ewm(span=26, adjust=False).mean()
    full_data["MACD"] = ema12 - ema26
    full_data["Signal_Line"] = full_data["MACD"].ewm(span=9, adjust=False).mean()

    # Bollinger Bands & ATR
    std20 = full_data["Close"].rolling(20).std()
    full_data["BB_Upper"] = full_data["MA20"] + (std20 * 2)
    full_data["BB_Lower"] = full_data["MA20"] - (std20 * 2)

    high_low = full_data["High"] - full_data["Low"]
    high_close = np.abs(full_data["High"] - full_data["Close"].shift())
    low_close = np.abs(full_data["Low"] - full_data["Close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    full_data["ATR"] = tr.rolling(14).mean().bfill()

    full_data = full_data.bfill().ffill()

    # Daily Volatility
    full_data["Daily_Range_Pct"] = ((full_data["High"] - full_data["Low"]) / full_data["Close"]) * 100
    avg_1day_volatility_pct = float(full_data["Daily_Range_Pct"].tail(20).mean())

    # Data Slice for Selected Horizon
    data_horizon = full_data.tail(max(trading_days_count, 5)).copy()
    current_price = float(data_horizon["Close"].iloc[-1])
    ma20 = float(data_horizon["MA20"].iloc[-1])
    ma50 = float(data_horizon["MA50"].iloc[-1])
    ma100 = float(data_horizon["MA100"].iloc[-1])
    bb_lower = float(data_horizon["BB_Lower"].iloc[-1])
    rsi = float(data_horizon["RSI"].iloc[-1])
    macd = float(data_horizon["MACD"].iloc[-1])
    signal_line = float(data_horizon["Signal_Line"].iloc[-1])
    atr = float(data_horizon["ATR"].iloc[-1])
    current_vol = float(data_horizon["Volume"].iloc[-1])
    vol_ma20 = float(data_horizon["Vol_MA20"].iloc[-1])

    # ==========================================
    # NEW: CURRENT DAY EXPECTED HIGH & LOW
    # ==========================================
    if len(full_data) >= 2:
        prev_high = float(full_data["High"].iloc[-2])
        prev_low = float(full_data["Low"].iloc[-2])
        prev_close = float(full_data["Close"].iloc[-2])
    else:
        prev_high, prev_low, prev_close = current_price, current_price, current_price

    # Pivot Point Method (Classic)
    pivot_point = (prev_high + prev_low + prev_close) / 3.0
    possible_high_pivot = (2 * pivot_point) - prev_low    # R1 Resistance
    possible_low_pivot = (2 * pivot_point) - prev_high     # S1 Support

    # ATR-based Expected Range
    possible_high_atr = current_price + (1.0 * atr)
    possible_low_atr = max(0, current_price - (1.0 * atr))

    # Combined Intraday Projections (Weighted Average)
    expected_day_high = (possible_high_pivot + possible_high_atr) / 2
    expected_day_low = (possible_low_pivot + possible_low_atr) / 2

    returns_slice = data_horizon["Close"].pct_change().dropna()
    positive_returns = returns_slice[returns_slice > 0]
    avg_horizon_growth_pct = (positive_returns.mean() * 100) if len(positive_returns) > 0 else 0

    # ==========================================
    # SAFE DIP ENTRY & TARGET CALCULATIONS
    # ==========================================
    investment = 42000
    cmp_shares = int(investment // current_price) if current_price > 0 else 0
    cmp_capital_used = cmp_shares * current_price

    cmp_target_price_1000 = current_price + (1000 / cmp_shares) if cmp_shares > 0 else current_price
    req_move_pct = ((cmp_target_price_1000 - current_price) / current_price) * 100 if current_price > 0 else 0

    # SAFE DIP PRICE LOGIC (Enhanced with RSI)
    if rsi > 70:
        safe_entry_price = ma50 if current_price > ma50 else bb_lower
        rsi_entry_eval = "Overbought (>70). High risk of immediate pullback. Wait for deeper structural support."
    elif rsi < 30:
        safe_entry_price = current_price
        rsi_entry_eval = "Oversold (<30). Price is already stretched downward; excellent momentum for a rebound."
    elif 30 <= rsi <= 45:
        safe_entry_price = max(bb_lower, current_price * 0.98)
        rsi_entry_eval = "Approaching Oversold. Good dip entry zone near structural support limits."
    else:
        if current_price > ma20:
            safe_entry_price = ma20
            rsi_entry_eval = "Neutral/Bullish. Entering at the 20-Day MA pullback is the most logical support."
        else:
            safe_entry_price = max(bb_lower, current_price * 0.98)
            rsi_entry_eval = "Neutral. Price is below MA20, lower Bollinger Band provides the next safe entry."

    safe_shares = int(investment // safe_entry_price) if safe_entry_price > 0 else cmp_shares
    safe_target_price_1000 = safe_entry_price + (1000 / safe_shares) if safe_shares > 0 else cmp_target_price_1000
    safe_dip_discount_pct = ((current_price - safe_entry_price) / current_price) * 100

    stop_loss_price = max(0, safe_entry_price - (1.5 * atr))

    # TIMEFRAME RECOMMENDATION
    if avg_1day_volatility_pct > 0:
        est_days_needed = int(np.ceil(req_move_pct / (avg_1day_volatility_pct * 0.5)))
    else:
        est_days_needed = 10

    if est_days_needed <= 1:
        recommended_timeframe = "1 Day (Tomorrow Focus)"
    elif est_days_needed <= 5:
        recommended_timeframe = "1 Week Focus"
    elif est_days_needed <= 10:
        recommended_timeframe = "2 Weeks Focus"
    elif est_days_needed <= 21:
        recommended_timeframe = "1 Month Focus"
    elif est_days_needed <= 63:
        recommended_timeframe = "3 Months Focus"
    else:
        recommended_timeframe = "6 Months / 1 Year Focus"

    # ==========================
    # 3. NEWS PROCESSING
    # ==========================
    news_list = ticker.news
    bullish_words = ["buy", "profit", "rise", "growth", "boost", "charter", "tender", "expand", "record", "gain"]
    bearish_words = ["war", "blockade", "cost", "risk", "fall", "decline", "probe", "loss", "penalty", "cut", "crisis", "scam"]

    news_summaries = []
    bullish_count, bearish_count = 0, 0

    if news_list:
        for item in news_list[:6]:
            content = item.get("content", {})
            title = content.get("title", "") if isinstance(content, dict) else item.get("title", "")
            summary = content.get("summary", title) if isinstance(content, dict) else title
            title_lower = str(title).lower() + " " + str(summary).lower()

            is_bullish = any(w in title_lower for w in bullish_words)
            is_bearish = any(w in title_lower for w in bearish_words)

            if is_bearish and not is_bullish:
                impact, color = "BEARISH", "red"
                bearish_count += 1
            elif is_bullish and not is_bearish:
                impact, color = "BULLISH", "green"
                bullish_count += 1
            else:
                impact, color = "NEUTRAL", "orange"

            news_summaries.append({"headline": title, "summary": summary, "impact": impact, "color": color})

    # ==========================
    # 4. TRADE SCORING
    # ==========================
    trend_score = 0
    if current_price > ma20: trend_score += 10
    if ma20 > ma50: trend_score += 10
    if ma50 > ma100: trend_score += 10

    rsi_score = 0
    if 45 <= rsi <= 60:
        rsi_score = 20
    elif 30 <= rsi < 45:
        rsi_score = 15
    elif 60 < rsi <= 70:
        rsi_score = 10
    elif rsi < 30:
        rsi_score = 15
    elif 70 < rsi <= 80:
        rsi_score = -10
    else:
        rsi_score = -25

    macd_score = 15 if macd > signal_line else 0

    feasibility_score = 0
    if avg_1day_volatility_pct > 0:
        ratio = req_move_pct / avg_1day_volatility_pct
        if ratio <= 0.6:
            feasibility_score = 25
        elif ratio <= 1.0:
            feasibility_score = 20
        elif ratio <= 1.5:
            feasibility_score = 10
        else:
            feasibility_score = 0

    volume_score = 10 if current_vol >= vol_ma20 else 5

    if bullish_count > bearish_count:
        news_score = 10
    elif bearish_count > bullish_count:
        news_score = -15
    else:
        news_score = 5

    raw_score = trend_score + rsi_score + macd_score + feasibility_score + volume_score + news_score
    trade_score = int(max(0, min(100, raw_score)))

    # ==========================
    # 5. DASHBOARD UI LAYOUT
    # ==========================
    st.markdown(f"### Stock: **{symbol}** | Horizon Target: **{selected_timeframe_label}**")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Current Market Price", f"₹{current_price:.2f}")
    col2.metric("RSI (14)", f"{rsi:.2f}")
    col3.metric("MACD Status", "BULLISH" if macd > signal_line else "BEARISH")
    col4.metric("Trade Score", f"{trade_score}/100")

    if trade_score >= 80:
        col5.success("EXCELLENT BUY")
    elif trade_score >= 65:
        col5.info("MODERATE BUY")
    elif trade_score >= 45:
        col5.warning("HOLD / NEUTRAL")
    else:
        col5.error("AVOID / HIGH RISK")

    st.markdown("---")

    # NEW SECTION: CURRENT DAY POSSIBLE HIGH & LOW
    st.subheader("📊 Today's Expected Price Range (Current Day High / Low Projections)")
    h_col1, h_col2, h_col3, h_col4 = st.columns(4)
    
    h_col1.metric("Expected Day High", f"₹{expected_day_high:.2f}", f"+{((expected_day_high - current_price)/current_price)*100:.2f}%")
    h_col2.metric("Expected Day Low", f"₹{expected_day_low:.2f}", f"{((expected_day_low - current_price)/current_price)*100:.2f}%")
    h_col3.metric("Pivot Point (P)", f"₹{pivot_point:.2f}")
    h_col4.metric("14-Day ATR Range", f"±₹{atr:.2f}")

    st.markdown("---")

    # SAFE ENTRY RECOMMENDATION BOX
    st.subheader("🛡️ Safe Dip Entry & Risk-Free Price Point")

    entry_col1, entry_col2, entry_col3, entry_col4 = st.columns(4)
    entry_col1.metric("Current Price (CMP)", f"₹{current_price:.2f}")
    entry_col2.metric("SAFE DIP BUY PRICE", f"₹{safe_entry_price:.2f}", f"-{safe_dip_discount_pct:.2f}% Dip")
    entry_col3.metric("Target Exit (₹1,000 Profit)", f"₹{safe_target_price_1000:.2f}")
    entry_col4.metric("Strict Stop Loss", f"₹{stop_loss_price:.2f}")
    
    st.caption(f"📈 **RSI Entry Evaluation:** {rsi_entry_eval} (Current RSI: {rsi:.2f})")

    if current_price <= (safe_entry_price * 1.005):
        st.success(
            f"🎯 **PRICE IS AT A SAFE ENTRY POINT!** Current Market Price (₹{current_price:.2f}) is already close to or at the Safe Dip Entry level (₹{safe_entry_price:.2f})."
        )
    else:
        st.info(
            f"💡 **WAIT FOR DIP OR PLACE LIMIT ORDER**: Current Price (₹{current_price:.2f}) is slightly higher than support. "
            f"Placing a buy limit order near **₹{safe_entry_price:.2f}** gives a safer trade setup to guarantee a **₹1,000 profit**."
        )

    st.markdown("---")

    # PROFIT FEASIBILITY & HORIZON ADVISORY BOX
    st.subheader("🎯 ₹1,000 Profit Horizon Analysis (₹42,000 Capital)")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Capital Allocated", f"₹{investment:,}")
    col_b.metric("Required Price Surge", f"+{req_move_pct:.2f}%", f"Target at CMP: ₹{cmp_target_price_1000:.2f}")
    col_c.metric("Avg Daily Volatility", f"{avg_1day_volatility_pct:.2f}%")

    if trading_days_count >= est_days_needed:
        st.success(
            f"✅ **SELECTED HORIZON IS SUFFICIENT!**\n\n"
            f"Your active timeframe ({selected_timeframe_label}) is well suited. Based on daily volatility ({avg_1day_volatility_pct:.2f}%/day), "
            f"the required **+{req_move_pct:.2f}%** surge for a **₹1,000 profit** is realistic within ~{est_days_needed} trading day(s)."
        )
    else:
        st.warning(
            f"⚠️ **RECOMMENDATION: INCREASE YOUR TIMEFRAME HORIZON!**\n\n"
            f"Reaching a ₹1,000 profit from CMP requires a **+{req_move_pct:.2f}%** move. "
            f"Because this stock moves ~{avg_1day_volatility_pct:.2f}% per day, **{selected_timeframe_label} is too short**.\n\n"
            f"👉 **Suggested Horizon**: Switch your dropdown to **{recommended_timeframe}** (~{est_days_needed} trading days) or enter at the **Safe Dip Price (₹{safe_entry_price:.2f})**."
        )

    st.markdown("---")

    left_col, right_col = st.columns([1.1, 0.9])

    with left_col:
        st.subheader("📊 Key Moving Averages")
        m1, m2, m3 = st.columns(3)
        m1.metric("MA 20 (1M)", f"₹{ma20:.2f}")
        m2.metric("MA 50 (2.5M)", f"₹{ma50:.2f}")
        m3.metric("MA 100 (5M)", f"₹{ma100:.2f}")

    with right_col:
        st.subheader("📰 News Sentiment & Market Headlines")
        if bearish_count > bullish_count + 1:
            st.error(
                f"⚠️ Outlook: Near-term headwinds from costs/geopolitics may create margin pressure over {selected_timeframe_label}.")
        elif bullish_count > bearish_count:
            st.success(
                f"🚀 Outlook: Strong momentum expected to push prices toward targets during {selected_timeframe_label}.")
        else:
            st.info(f"➡️ Outlook: Sideways-to-mild upward movement expected for {selected_timeframe_label}.")

        if news_summaries:
            for item in news_summaries:
                with st.expander(f":{item['color']}[[{item['impact']}]] {item['headline']}"):
                    st.write(item["summary"])
        else:
            st.write("No recent news stories found for this symbol.")

    # Interactive Chart with Subplots for RSI
    st.markdown("---")
    
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03, 
        row_heights=[0.7, 0.3],
        subplot_titles=(f"📉 Technical Chart ({selected_timeframe_label})", "RSI (14)")
    )

    # 1. Main Price Chart
    fig.add_trace(go.Candlestick(
        x=data_horizon.index, open=data_horizon["Open"], high=data_horizon["High"],
        low=data_horizon["Low"], close=data_horizon["Close"], name="Price"
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=data_horizon.index, y=data_horizon["MA20"], mode="lines", name="MA20", line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=data_horizon.index, y=data_horizon["MA50"], mode="lines", name="MA50", line=dict(color='orange')), row=1, col=1)
    fig.add_trace(go.Scatter(x=data_horizon.index, y=data_horizon["BB_Upper"], mode="lines", name="BB Upper", line=dict(dash='dash', color='gray')), row=1, col=1)
    fig.add_trace(go.Scatter(x=data_horizon.index, y=data_horizon["BB_Lower"], mode="lines", name="BB Lower", line=dict(dash='dash', color='gray')), row=1, col=1)

    # Add Horizontal lines on chart for Expected Day High and Low
    fig.add_hline(y=expected_day_high, line_dash="dot", line_color="lime", row=1, col=1, 
                  annotation_text=f"Exp High (₹{expected_day_high:.2f})", annotation_position="top left")
    fig.add_hline(y=expected_day_low, line_dash="dot", line_color="crimson", row=1, col=1, 
                  annotation_text=f"Exp Low (₹{expected_day_low:.2f})", annotation_position="bottom left")

    # 2. RSI Subplot
    fig.add_trace(go.Scatter(
        x=data_horizon.index, y=data_horizon["RSI"], mode="lines", name="RSI", line=dict(color='purple')
    ), row=2, col=1)

    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1, annotation_text="Overbought (70)", annotation_position="bottom right")
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1, annotation_text="Oversold (30)", annotation_position="top right")

    fig.update_layout(
        template="plotly_dark", 
        height=700, 
        xaxis_rangeslider_visible=False,
        xaxis2_rangeslider_visible=False,
        showlegend=True,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)
