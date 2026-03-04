//+------------------------------------------------------------------+
//| AlmostFinishedBot_EA.mq5                                        |
//| v1.0 - Regime-aware | Kelly sizing | SMC logic | Multi-TF       |
//| Fallback: EMA cross + RSI when ML unavailable                   |
//+------------------------------------------------------------------+
#property copyright "AlmostFinishedBot"
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>
CTrade Trade;

// ── Input parameters ───────────────────────────────────────────
input string   Symbols       = "XAUUSD,XAUSGD";
input double   RiskPercent   = 1.0;
input int      MaxTrades     = 2;
input double   MaxDrawdownPct= 10.0;
input double   MaxSpreadATR  = 0.5;
input bool     UseTrailing   = true;
input bool     UseSMC        = true;    // Smart Money Concept filter
input bool     UseKelly      = true;    // Kelly Criterion sizing
input bool     UseRegime     = true;    // Market regime filter
input bool     SwingMode     = true;    // Swing/Sniper dual mode
input string   TelegramToken = "";
input long     TelegramChatID= 838489368;
input int      MagicNumber   = 20250303;

// ── Globals ───────────────────────────────────────────────────
string   sym_list[];
bool     ml_warned = false;
datetime last_signal_time = 0;
double   session_start_balance = 0;
double   profit_buffer = 0;   // Tier 3: Sniper mode builds this
int      day_trades = 0;
bool     swing_active = false;
int      swing_ticket = 0;

//+------------------------------------------------------------------+
int OnInit()
{
   Print("AlmostFinishedBot EA v1.0 started | Balance: $", AccountInfoDouble(ACCOUNT_BALANCE));
   StringSplit(Symbols, ',', sym_list);
   Print("Symbols: ", Symbols, " | Risk: ", RiskPercent, "% | MaxTrades: ", MaxTrades);
   session_start_balance = AccountInfoDouble(ACCOUNT_BALANCE);

   if(SymbolInfoDouble(_Symbol, SYMBOL_BID) <= 0)
      Print("WARNING: Symbol ", _Symbol, " may not be available");

   SendTG("AlmostFinishedBot EA started on " + _Symbol);
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   Print("AlmostFinishedBot EA OnDeinit called");
}

//+------------------------------------------------------------------+
void OnTick()
{
   // ── Drawdown guard ─────────────────────────────────────────────
   double bal = AccountInfoDouble(ACCOUNT_BALANCE);
   double eq  = AccountInfoDouble(ACCOUNT_EQUITY);
   if(session_start_balance > 0)
   {
      double dd_pct = (session_start_balance - eq) / session_start_balance * 100.0;
      if(dd_pct >= MaxDrawdownPct)
      {
         Print("MAX DRAWDOWN REACHED: ", dd_pct, "% — halting trading");
         CloseAll();
         return;
      }
   }

   // ── Max trades guard ───────────────────────────────────────────
   if(CountOpenTrades() >= MaxTrades) return;

   // Only execute on new bar
   static datetime last_bar = 0;
   datetime this_bar = iTime(_Symbol, PERIOD_CURRENT, 0);
   if(this_bar == last_bar) return;
   last_bar = this_bar;

   // ── Process each symbol ────────────────────────────────────────
   for(int i = 0; i < ArraySize(sym_list); i++)
   {
      string sym = StringTrimLeft(StringTrimRight(sym_list[i]));
      if(sym == "") continue;
      if(SymbolInfoDouble(sym, SYMBOL_BID) <= 0)
      {
         Print("Symbol unavailable: ", sym, " — skipping");
         continue;
      }
      ProcessSymbol(sym);
   }

   // ── Trail existing positions ───────────────────────────────────
   if(UseTrailing) TrailPositions();

   // ── Update profit buffer (Sniper mode) ─────────────────────────
   UpdateProfitBuffer();
}

//+------------------------------------------------------------------+
void ProcessSymbol(string sym)
{
   // ── Spread filter ──────────────────────────────────────────────
   double spread = SymbolInfoInteger(sym, SYMBOL_SPREAD) * SymbolInfoDouble(sym, SYMBOL_POINT);
   double atr    = GetATR(sym, PERIOD_CURRENT, 14);
   if(atr > 0 && spread > atr * MaxSpreadATR) return;

   // ── Market Regime filter (Tier 1) ──────────────────────────────
   string regime = "UNKNOWN";
   if(UseRegime) regime = GetRegime(sym);
   // In LOW_VOL regime: reduce lot size / skip
   if(regime == "LOW_VOL") return;

   // ── Multi-TF analysis: Higher TF direction (Tier 1) ────────────
   int htf_direction = GetHTFDirection(sym);  // +1 BUY, -1 SELL, 0 neutral

   // ── SMC: Order Block + Liquidity check (Tier 2) ────────────────
   bool smc_ok = true;
   if(UseSMC) smc_ok = CheckSMC(sym);

   // ── Read ML signal ─────────────────────────────────────────────
   int ml_signal = 0; double ml_conf = 0.5; double kelly_frac = 0.01;
   bool ml_ok = ReadMLSignal(ml_signal, ml_conf, kelly_frac);

   // ── Fallback EMA + RSI ─────────────────────────────────────────
   int signal = 0;
   if(!ml_ok)
   {
      if(!ml_warned) { Print("INFO: ML signal not found — using technical fallback"); ml_warned=true; }
      signal = FallbackSignal(sym);
      ml_conf = 0.55; kelly_frac = RiskPercent / 100.0;
   }
   else
   {
      signal = ml_signal;
   }

   if(signal == 0) return;

   // ── Confirm with HTF ───────────────────────────────────────────
   if(htf_direction != 0 && htf_direction != signal) return;

   // ── SMC filter ─────────────────────────────────────────────────
   if(UseSMC && !smc_ok) return;

   // ── Regime-based signal filter ─────────────────────────────────
   // In RANGING market: avoid trend-following signals from ML
   if(regime == "RANGING" && !ml_ok) return;

   // ── Kelly / Confidence-based position sizing (Tier 3) ──────────
   double risk_pct = RiskPercent;
   if(UseKelly && ml_ok)
   {
      double kelly_pct = kelly_frac * 100.0;
      // Scale with confidence
      if(ml_conf > 0.80)      risk_pct = kelly_pct * 2.0;
      else if(ml_conf > 0.70) risk_pct = kelly_pct * 1.5;
      else if(ml_conf > 0.60) risk_pct = kelly_pct * 1.0;
      else                    risk_pct = kelly_pct * 0.5;
      risk_pct = MathMin(risk_pct, 3.0);  // Hard cap 3%
   }

   // ── Swing mode: only use profit buffer for swing trades ────────
   if(SwingMode && swing_active) return;  // Already in a swing trade

   // ── Calculate SL/TP ────────────────────────────────────────────
   double sl_pts = atr * 2.0;  // 2× ATR stop
   double tp_pts = atr * 3.0;  // 3× ATR target (1:3 R:R)

   // ── Lot calculation with Kelly ─────────────────────────────────
   double lots = CalcLots(sym, sl_pts, risk_pct);
   if(lots <= 0) return;

   // ── Place order ────────────────────────────────────────────────
   double bid = SymbolInfoDouble(sym, SYMBOL_BID);
   double ask = SymbolInfoDouble(sym, SYMBOL_ASK);
   double pt  = SymbolInfoDouble(sym, SYMBOL_POINT);

   double sl, tp;
   if(signal == 1)  // BUY
   {
      sl = ask - sl_pts;
      tp = ask + tp_pts;
      if(Trade.Buy(lots, sym, ask, sl, tp))
      {
         string msg = sym + " BUY | Lots: " + DoubleToString(lots,2) +
                      " | SL: " + DoubleToString(sl_pts/pt,1) +
                      " | TP: " + DoubleToString(tp_pts/pt,1) +
                      " | ATR: " + DoubleToString(atr/pt,1) +
                      " | Conf: " + DoubleToString(ml_conf*100,1) + "%" +
                      " | Regime: " + regime +
                      " | Risk: " + DoubleToString(risk_pct,2) + "%";
         Print(msg); SendTG(msg);
         if(SwingMode && ml_conf > 0.75) { swing_active=true; swing_ticket=(int)Trade.ResultOrder(); }
      }
   }
   else if(signal == -1)  // SELL
   {
      sl = bid + sl_pts;
      tp = bid - tp_pts;
      if(Trade.Sell(lots, sym, bid, sl, tp))
      {
         string msg = sym + " SELL | Lots: " + DoubleToString(lots,2) +
                      " | SL: " + DoubleToString(sl_pts/pt,1) +
                      " | TP: " + DoubleToString(tp_pts/pt,1) +
                      " | ATR: " + DoubleToString(atr/pt,1) +
                      " | Conf: " + DoubleToString(ml_conf*100,1) + "%" +
                      " | Regime: " + regime +
                      " | Risk: " + DoubleToString(risk_pct,2) + "%";
         Print(msg); SendTG(msg);
         if(SwingMode && ml_conf > 0.75) { swing_active=true; swing_ticket=(int)Trade.ResultOrder(); }
      }
   }
}

//+------------------------------------------------------------------+
// Fallback: EMA cross + RSI + MACD confirmation
int FallbackSignal(string sym)
{
   double e20=GetEMA(sym,PERIOD_CURRENT,20,0), e50=GetEMA(sym,PERIOD_CURRENT,50,0);
   double e20p=GetEMA(sym,PERIOD_CURRENT,20,1),e50p=GetEMA(sym,PERIOD_CURRENT,50,1);
   double rsi=GetRSI(sym,PERIOD_CURRENT,14,0);

   bool ema_cross_up  = (e20>e50) && (e20p<=e50p);
   bool ema_cross_dn  = (e20<e50) && (e20p>=e50p);
   // RSI confirmation: not overbought/oversold
   if(ema_cross_up  && rsi>40 && rsi<70) return  1;
   if(ema_cross_dn  && rsi>30 && rsi<60) return -1;
   return 0;
}

//+------------------------------------------------------------------+
// Market Regime via ADX (simple inline version)
string GetRegime(string sym)
{
   // Get ADX value
   int adx_handle = iADX(sym, PERIOD_CURRENT, 14);
   if(adx_handle == INVALID_HANDLE) return "UNKNOWN";
   double adx_buf[1];
   if(CopyBuffer(adx_handle, 0, 0, 1, adx_buf) <= 0) return "UNKNOWN";
   double adx = adx_buf[0];
   IndicatorRelease(adx_handle);

   // Get Bollinger width
   int bb_handle = iBands(sym, PERIOD_CURRENT, 20, 0, 2.0, PRICE_CLOSE);
   if(bb_handle == INVALID_HANDLE) return "UNKNOWN";
   double upper[1], lower[1], mid[1];
   CopyBuffer(bb_handle, 1, 0, 1, upper);
   CopyBuffer(bb_handle, 2, 0, 1, lower);
   CopyBuffer(bb_handle, 0, 0, 1, mid);
   IndicatorRelease(bb_handle);
   double bbw = (mid[0]>0) ? (upper[0]-lower[0])/mid[0] : 0;

   if(adx >= 40 && bbw >= 0.04) return "TRENDING_STRONG";
   if(adx >= 25) return "TRENDING";
   if(bbw >= 0.04) return "HIGH_VOL";
   if(bbw <= 0.015) return "LOW_VOL";
   return "RANGING";
}

//+------------------------------------------------------------------+
// Higher Timeframe direction (Tier 1: Multi-TF)
int GetHTFDirection(string sym)
{
   ENUM_TIMEFRAMES htf = PERIOD_H4;
   double ema20 = GetEMA(sym, htf, 20, 0);
   double ema50 = GetEMA(sym, htf, 50, 0);
   double close = iClose(sym, htf, 0);
   if(close > ema20 && ema20 > ema50) return  1;
   if(close < ema20 && ema20 < ema50) return -1;
   return 0;
}

//+------------------------------------------------------------------+
// SMC: Simple Order Block / Liquidity check (Tier 2)
bool CheckSMC(string sym)
{
   // Check for recent swing high/low taken out (liquidity sweep)
   double high_5 = 0, low_5 = 99999999;
   for(int i = 2; i <= 6; i++)
   {
      high_5 = MathMax(high_5, iHigh(sym, PERIOD_CURRENT, i));
      low_5  = MathMin(low_5,  iLow(sym,  PERIOD_CURRENT, i));
   }
   double curr_high = iHigh(sym, PERIOD_CURRENT, 1);
   double curr_low  = iLow(sym,  PERIOD_CURRENT, 1);
   double curr_close= iClose(sym,PERIOD_CURRENT, 1);

   // Bullish SMC: swept below swing low then closed above
   bool bull_sweep = (curr_low < low_5) && (curr_close > low_5);
   // Bearish SMC: swept above swing high then closed below
   bool bear_sweep = (curr_high > high_5) && (curr_close < high_5);

   return (bull_sweep || bear_sweep);
}

//+------------------------------------------------------------------+
// Read ML signal from JSON file
bool ReadMLSignal(int &signal, double &conf, double &kelly)
{
   string path = "AlmostFinishedBot\\current_signal.json";
   int fh = FileOpen(path, FILE_READ|FILE_TXT|FILE_ANSI);
   if(fh == INVALID_HANDLE) return false;
   string content = "";
   while(!FileIsEnding(fh)) content += FileReadString(fh);
   FileClose(fh);

   // Simple JSON parse
   string sig_str = JSONGetString(content, "signal");
   conf  = StringToDouble(JSONGetString(content, "confidence"));
   kelly = StringToDouble(JSONGetString(content, "kelly_fraction"));
   if(sig_str == "BUY")  signal =  1;
   else if(sig_str == "SELL") signal = -1;
   else signal = 0;
   return (sig_str != "");
}

string JSONGetString(string json, string key)
{
   string search = "\"" + key + "\":";
   int pos = StringFind(json, search);
   if(pos < 0) return "";
   pos += StringLen(search);
   // Skip whitespace
   while(pos < StringLen(json) && StringGetCharacter(json, pos) == ' ') pos++;
   // Handle string value
   if(StringGetCharacter(json, pos) == '"')
   {
      pos++;
      string result = "";
      while(pos < StringLen(json) && StringGetCharacter(json, pos) != '"')
      {
         result += ShortToString(StringGetCharacter(json, pos)); pos++;
      }
      return result;
   }
   // Handle numeric value
   string result = "";
   while(pos < StringLen(json))
   {
      ushort ch = StringGetCharacter(json, pos);
      if(ch==','||ch=='}'||ch=='\n'||ch=='\r') break;
      result += ShortToString(ch); pos++;
   }
   return StringTrimRight(StringTrimLeft(result));
}

//+------------------------------------------------------------------+
// Kelly-aware lot calculation
double CalcLots(string sym, double sl_pts, double risk_pct)
{
   double bal       = AccountInfoDouble(ACCOUNT_BALANCE);
   double risk_amt  = bal * (risk_pct / 100.0);
   double tick_val  = SymbolInfoDouble(sym, SYMBOL_TRADE_TICK_VALUE);
   double tick_size = SymbolInfoDouble(sym, SYMBOL_TRADE_TICK_SIZE);
   double min_lot   = SymbolInfoDouble(sym, SYMBOL_VOLUME_MIN);
   double max_lot   = MathMin(SymbolInfoDouble(sym, SYMBOL_VOLUME_MAX), 0.50);
   double lot_step  = SymbolInfoDouble(sym, SYMBOL_VOLUME_STEP);

   if(tick_val <= 0 || sl_pts <= 0) return min_lot;

   double lots = risk_amt / (sl_pts / tick_size * tick_val);
   lots = MathMax(min_lot, MathMin(max_lot, lots));
   lots = MathFloor(lots / lot_step) * lot_step;

   // Margin check
   double margin_needed;
   if(!OrderCalcMargin(ORDER_TYPE_BUY, sym, lots,
                       SymbolInfoDouble(sym,SYMBOL_ASK), margin_needed))
      return min_lot;
   if(margin_needed > bal * 0.20)
   {
      lots = MathFloor((bal*0.20/margin_needed) * lots / lot_step) * lot_step;
      lots = MathMax(min_lot, lots);
   }
   return lots;
}

//+------------------------------------------------------------------+
// ATR-based trailing stop (Tier 4: Multi-TF Trail)
void TrailPositions()
{
   double atr_h4 = GetATR(_Symbol, PERIOD_H4, 14);  // Higher TF ATR trail
   double trail  = atr_h4 * 1.5;

   for(int i = PositionsTotal()-1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetInteger(POSITION_MAGIC) != MagicNumber) continue;
      string sym = PositionGetString(POSITION_SYMBOL);
      double bid = SymbolInfoDouble(sym, SYMBOL_BID);
      double ask = SymbolInfoDouble(sym, SYMBOL_ASK);
      long   ptype = PositionGetInteger(POSITION_TYPE);
      double cur_sl = PositionGetDouble(POSITION_SL);
      if(trail <= 0) continue;

      if(ptype == POSITION_TYPE_BUY)
      {
         double new_sl = bid - trail;
         if(new_sl > cur_sl + SymbolInfoDouble(sym,SYMBOL_POINT))
            Trade.PositionModify(ticket, new_sl, PositionGetDouble(POSITION_TP));
      }
      else if(ptype == POSITION_TYPE_SELL)
      {
         double new_sl = ask + trail;
         if(new_sl < cur_sl - SymbolInfoDouble(sym,SYMBOL_POINT) || cur_sl == 0)
            Trade.PositionModify(ticket, new_sl, PositionGetDouble(POSITION_TP));
      }
   }
}

//+------------------------------------------------------------------+
void UpdateProfitBuffer()
{
   if(!SwingMode) return;
   double cur_equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double profit = cur_equity - session_start_balance;
   if(profit > 0) profit_buffer = profit;

   // Reset swing if trade closed
   if(swing_active && swing_ticket > 0)
   {
      if(!PositionSelectByTicket(swing_ticket))
      { swing_active=false; swing_ticket=0; }
   }
}

//+------------------------------------------------------------------+
int CountOpenTrades()
{
   int count = 0;
   for(int i = 0; i < PositionsTotal(); i++)
      if(PositionGetInteger(POSITION_MAGIC) == MagicNumber) count++;
   return count;
}

void CloseAll()
{
   for(int i = PositionsTotal()-1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(PositionGetInteger(POSITION_MAGIC) == MagicNumber)
         Trade.PositionClose(ticket);
   }
}

//+------------------------------------------------------------------+
double GetEMA(string sym, ENUM_TIMEFRAMES tf, int period, int shift)
{
   int h = iMA(sym, tf, period, 0, MODE_EMA, PRICE_CLOSE);
   if(h == INVALID_HANDLE) return 0;
   double buf[1];
   if(CopyBuffer(h, 0, shift, 1, buf) <= 0) return 0;
   IndicatorRelease(h); return buf[0];
}

double GetRSI(string sym, ENUM_TIMEFRAMES tf, int period, int shift)
{
   int h = iRSI(sym, tf, period, PRICE_CLOSE);
   if(h == INVALID_HANDLE) return 50;
   double buf[1];
   if(CopyBuffer(h, 0, shift, 1, buf) <= 0) return 50;
   IndicatorRelease(h); return buf[0];
}

double GetATR(string sym, ENUM_TIMEFRAMES tf, int period)
{
   int h = iATR(sym, tf, period);
   if(h == INVALID_HANDLE) return 0;
   double buf[1];
   if(CopyBuffer(h, 0, 0, 1, buf) <= 0) return 0;
   IndicatorRelease(h); return buf[0];
}

//+------------------------------------------------------------------+
void SendTG(string msg)
{
   if(TelegramToken == "" || TelegramChatID == 0) return;
   string url = "https://api.telegram.org/bot" + TelegramToken +
                "/sendMessage?chat_id=" + IntegerToString(TelegramChatID) +
                "&text=" + msg;
   char   req[], res[]; string headers;
   ArrayResize(req, 0);
   int r = WebRequest("GET", url, "", "", 3000, req, 0, res, headers);
   if(r > 0) Print("Telegram sent, response length: ", ArraySize(res));
}
