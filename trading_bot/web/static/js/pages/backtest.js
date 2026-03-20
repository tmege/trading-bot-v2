(function() {
  'use strict';
  window.TB = window.TB || {};
  TB.pages = TB.pages || {};

  let intervals = [];
  let progressSSE = null;
  let equityChart = null;
  let equitySeries = [];
  let strategyMap = {}; // file -> coins[]
  let availableCoins = {}; // coin -> {years_available, min_date, max_date}

  function mount(container) {
    container.innerHTML = `
      <div class="section-header"><h2>Backtest</h2></div>
      <div class="card mb-16">
        <div class="card-header">Configuration</div>
        <div class="card-body">
          <div class="form-row mb-12">
            <div class="form-group" style="flex:2;">
              <label>Strategy</label>
              <select id="bt-strategy" class="input w-full"></select>
            </div>
            <div class="form-group" style="flex:1;">
              <label>Start Date</label>
              <input id="bt-start-date" type="date" class="input w-full">
            </div>
            <div class="form-group" style="flex:1;">
              <label>End Date</label>
              <input id="bt-end-date" type="date" class="input w-full">
            </div>
          </div>
          <div class="form-group mb-12">
            <label>Coins</label>
            <div id="bt-coins" class="pills"></div>
          </div>
          <div id="bt-date-info" class="c-text2" style="font-size:11px;margin-bottom:12px;"></div>
          <button class="btn btn-primary" id="btn-bt-run">Run Backtest</button>
        </div>
      </div>
      <div class="card mb-16" id="bt-progress-card" style="display:none;">
        <div class="card-header">Progress</div>
        <div class="card-body" id="bt-progress"></div>
      </div>
      <div id="bt-results" style="display:none;">
        <div class="card mb-16">
          <div class="card-header">Results Comparison</div>
          <div class="card-body overflow-auto bt-compare" id="bt-compare"></div>
        </div>
        <div class="card mb-16">
          <div class="card-header">Equity Curve (%)</div>
          <div class="card-body"><div id="bt-equity" class="chart-container"></div></div>
        </div>
        <div class="card mb-16" id="bt-detail-card" style="display:none;">
          <div class="card-header" id="bt-detail-header">Detailed Stats</div>
          <div class="card-body" id="bt-detail"></div>
        </div>
        <div class="card mb-16" id="bt-trades-card" style="display:none;">
          <div class="card-header">Trade Journal</div>
          <div class="card-body overflow-auto" style="max-height:400px;" id="bt-trades"></div>
          <div class="pagination" id="bt-trades-pag"></div>
        </div>
      </div>
      <div class="card">
        <div class="card-header">
          <span>History</span>
          <button class="btn btn-sm btn-danger" id="btn-bt-clear">Clear</button>
        </div>
        <div class="card-body overflow-auto" style="max-height:400px;" id="bt-history"></div>
      </div>
    `;

    loadConfig();
    loadHistory();

    document.getElementById('btn-bt-run').onclick = runBacktest;
    document.getElementById('btn-bt-clear').onclick = clearHistory;
  }

  function unmount() {
    intervals.forEach(clearInterval);
    intervals = [];
    if (progressSSE) { progressSSE.close(); progressSSE = null; }
    if (equityChart) { equityChart.remove(); equityChart = null; }
    equitySeries = [];
  }

  async function loadConfig() {
    const [strats, coins] = await Promise.all([
      TB.api.get('/api/strategies'),
      TB.api.get('/api/backtest/coins'),
    ]);

    // Build lookup: coin -> data availability
    availableCoins = {};
    if (coins) coins.forEach(c => { availableCoins[c.coin] = c; });

    // Build lookup: strategy file -> configured coins
    strategyMap = {};
    if (strats) strats.forEach(s => { strategyMap[s.file] = s.coins || []; });

    const sel = document.getElementById('bt-strategy');
    if (sel && strats) {
      sel.innerHTML = strats.map(s =>
        `<option value="${TB.utils.esc(s.file)}">${TB.utils.esc(s.name)} (${s.coins.map(c => TB.utils.esc(c)).join(', ')})</option>`
      ).join('');
      sel.onchange = () => { updateCoinsForStrategy(); loadLatestResults(); };
    }

    updateCoinsForStrategy();
    loadLatestResults();
  }

  function updateCoinsForStrategy() {
    const sel = document.getElementById('bt-strategy');
    const coinsEl = document.getElementById('bt-coins');
    const dateInfo = document.getElementById('bt-date-info');
    const startInput = document.getElementById('bt-start-date');
    const endInput = document.getElementById('bt-end-date');
    if (!sel || !coinsEl) return;

    const file = sel.value;
    const stratCoins = strategyMap[file] || [];

    // Find date range across all coins for this strategy
    let globalMin = '';
    let globalMax = '';
    stratCoins.forEach(c => {
      const info = availableCoins[c];
      if (info) {
        if (!globalMin || info.min_date < globalMin) globalMin = info.min_date;
        if (!globalMax || info.max_date > globalMax) globalMax = info.max_date;
      }
    });

    // Default: 6 months ago → today (clamped to available data)
    const today = new Date().toISOString().slice(0, 10);
    const sixMonthsAgo = new Date(Date.now() - 180 * 86400000).toISOString().slice(0, 10);

    if (startInput) {
      startInput.min = globalMin;
      startInput.max = globalMax;
      if (!startInput.value) {
        startInput.value = globalMin && sixMonthsAgo < globalMin ? globalMin : sixMonthsAgo;
      }
    }
    if (endInput) {
      endInput.min = globalMin;
      endInput.max = globalMax;
      if (!endInput.value) {
        endInput.value = globalMax && today > globalMax ? globalMax : today;
      }
    }

    if (dateInfo && globalMin && globalMax) {
      dateInfo.textContent = `Data available: ${globalMin} to ${globalMax}`;
    }

    coinsEl.innerHTML = stratCoins.map(c => {
      const info = availableCoins[c];
      const years = info ? info.years_available + 'y' : 'no data';
      const hasData = !!info;
      return `<span class="pill ${hasData ? 'active' : 'disabled'}" data-coin="${TB.utils.esc(c)}">
        ${TB.utils.esc(c)} <span class="c-text2" style="font-size:9px;">(${years})</span>
      </span>`;
    }).join('');
  }

  async function loadLatestResults() {
    const sel = document.getElementById('bt-strategy');
    if (!sel || !sel.value) return;

    const data = await TB.api.get('/api/backtest/latest/' + encodeURIComponent(sel.value));
    if (!data || data.error || !data.results) {
      document.getElementById('bt-results').style.display = 'none';
      return;
    }

    showResults(data.results);
  }

  async function runBacktest() {
    const strategy = document.getElementById('bt-strategy').value;
    const startDate = document.getElementById('bt-start-date').value;
    const endDate = document.getElementById('bt-end-date').value;
    const coinEls = document.querySelectorAll('#bt-coins .pill.active');
    const coins = Array.from(coinEls).map(el => el.dataset.coin);

    if (!strategy || coins.length === 0) {
      TB.toast.show('error', 'Select a strategy and at least one coin');
      return;
    }

    const result = await TB.api.post('/api/backtest/run', {
      strategy, coins, start_date: startDate, end_date: endDate,
    });

    if (!result || result.error) {
      TB.toast.show('error', result ? result.error : 'Failed to start backtest');
      return;
    }

    document.getElementById('bt-progress-card').style.display = '';
    document.getElementById('bt-results').style.display = 'none';
    document.getElementById('btn-bt-run').disabled = true;

    const progressEl = document.getElementById('bt-progress');
    progressEl.innerHTML = coins.map(c => {
      const sc = TB.utils.esc(c);
      return `<div class="mb-12"><div class="flex gap-8" style="align-items:center;"><strong>${sc}</strong><span id="bt-pct-${sc}" class="c-text2" style="font-size:11px;">0%</span></div>
       <div class="progress mt-8"><div id="bt-bar-${sc}" class="progress-bar" style="width:0%;"></div></div></div>`;
    }).join('');

    const allResults = {};

    progressSSE = TB.api.sse('/api/backtest/progress/' + result.run_id, (msg) => {
      if (msg.type === 'progress') {
        const bar = document.getElementById('bt-bar-' + msg.coin);
        const pct = document.getElementById('bt-pct-' + msg.coin);
        if (bar) bar.style.width = msg.pct + '%';
        if (pct) pct.textContent = msg.pct + '%';
      } else if (msg.type === 'coin_done') {
        const bar = document.getElementById('bt-bar-' + msg.coin);
        if (bar) { bar.style.width = '100%'; bar.classList.add('green'); }
        if (msg.result) allResults[msg.coin] = msg.result;
        if (msg.error) {
          TB.toast.show('warning', `${msg.coin}: ${msg.error}`);
        }
      } else if (msg.type === 'complete') {
        if (progressSSE) { progressSSE.close(); progressSSE = null; }
        document.getElementById('btn-bt-run').disabled = false;
        showResults(allResults);
        loadHistory();
      } else if (msg.type === 'error') {
        if (progressSSE) { progressSSE.close(); progressSSE = null; }
        document.getElementById('btn-bt-run').disabled = false;
        TB.toast.show('error', msg.message || 'Backtest failed');
      }
    });
  }

  function showResults(results) {
    document.getElementById('bt-results').style.display = '';
    const coins = Object.keys(results);
    if (coins.length === 0) return;

    // Comparison table — all values in %
    const compareEl = document.getElementById('bt-compare');
    const metrics = ['return_pct', 'sharpe_ratio', 'max_drawdown_pct', 'win_rate', 'total_trades', 'profit_factor'];
    const labels = ['Return %', 'Sharpe', 'Max DD %', 'Win Rate %', 'Trades', 'Profit Factor'];

    let bestReturn = Math.max(...coins.map(c => results[c].return_pct));

    compareEl.innerHTML = `<table class="tbl"><thead><tr><th>Metric</th>${coins.map(c =>
      `<th>${TB.utils.esc(c)}${results[c].return_pct === bestReturn && coins.length > 1 ? ' <span class="badge badge-green">BEST</span>' : ''}</th>`
    ).join('')}</tr></thead><tbody>
      ${metrics.map((m, i) => `<tr><td><strong>${labels[i]}</strong></td>${coins.map(c => {
        let v = results[c][m];
        if (m === 'profit_factor' && v >= 999) v = '\u221e';
        else if (typeof v === 'number') v = v.toFixed(2);
        const cls = (m === 'return_pct') ? TB.utils.pnlClass(results[c][m]) : '';
        return `<td class="mono ${cls}">${TB.utils.esc(String(v))}</td>`;
      }).join('')}</tr>`).join('')}
      <tr><td><strong>Verdict</strong></td>${coins.map(c =>
        `<td><span class="badge verdict-${TB.utils.esc(results[c].verdict || '')}">${TB.utils.esc(results[c].verdict || '\u2014')}</span></td>`
      ).join('')}</tr>
    </tbody></table>`;

    // Equity chart (in %)
    initEquityChart(results);

    // Detail for first coin
    if (coins.length > 0) showDetail(coins[0], results[coins[0]]);
  }

  function showDetail(coin, r) {
    const card = document.getElementById('bt-detail-card');
    const header = document.getElementById('bt-detail-header');
    const el = document.getElementById('bt-detail');
    if (!card || !el) return;
    card.style.display = '';
    if (header) header.textContent = `Stats: ${coin}`;

    const wf = r.walk_forward;
    const startBal = r.start_balance || 100;

    // Convert absolute values to % of balance
    const feesPct = (r.total_fees / startBal * 100).toFixed(2);
    const avgWinPct = (r.avg_win / startBal * 100).toFixed(2);
    const avgLossPct = (r.avg_loss / startBal * 100).toFixed(2);

    el.innerHTML = `<div class="grid-3">
      <div>
        <h4 class="mb-12">PnL</h4>
        <div class="metric-card mb-12"><div class="label">Return</div><div class="value ${TB.utils.pnlClass(r.return_pct)}">${TB.utils.fmtPct(r.return_pct)}</div></div>
        <div class="metric-card mb-12"><div class="label">Fees</div><div class="value">${feesPct}%</div></div>
        <div class="metric-card mb-12"><div class="label">Avg Win</div><div class="value c-green">${avgWinPct}%</div></div>
        <div class="metric-card"><div class="label">Avg Loss</div><div class="value c-red">${avgLossPct}%</div></div>
      </div>
      <div>
        <h4 class="mb-12">Risk</h4>
        <div class="metric-card mb-12"><div class="label">Sharpe</div><div class="value">${r.sharpe_ratio}</div></div>
        <div class="metric-card mb-12"><div class="label">Sortino</div><div class="value">${r.sortino_ratio}</div></div>
        <div class="metric-card mb-12"><div class="label">Max Drawdown</div><div class="value c-red">${r.max_drawdown_pct}%</div></div>
        <div class="metric-card mb-12"><div class="label">Win Rate</div><div class="value">${r.win_rate}%</div></div>
        <div class="metric-card"><div class="label">Profit Factor</div><div class="value">${r.profit_factor >= 999 ? '\u221e' : r.profit_factor}</div></div>
      </div>
      <div>
        <h4 class="mb-12">Walk-Forward</h4>
        ${wf ? `
          <div class="metric-card mb-12"><div class="label">IS Return</div><div class="value ${TB.utils.pnlClass(wf.is_return)}">${TB.utils.fmtPct(wf.is_return)}</div></div>
          <div class="metric-card mb-12"><div class="label">OOS Return</div><div class="value ${TB.utils.pnlClass(wf.oos_return)}">${TB.utils.fmtPct(wf.oos_return)}</div></div>
          <div class="metric-card mb-12"><div class="label">Decay</div><div class="value ${wf.decay_pct < -50 ? 'c-red' : ''}">${TB.utils.fmtPct(wf.decay_pct)}</div></div>
          ${wf.overfit_alert ? '<div class="alerts-banner" style="font-size:11px;">OVERFIT ALERT: Decay > 50%</div>' : ''}
        ` : '<div class="c-text2">Not enough data</div>'}
      </div>
    </div>`;

    // Trades journal
    showTradeJournal(r.trades || [], startBal);
  }

  let btTradeOffset = 0;
  let btTradesAll = [];
  let btStartBal = 100;

  function showTradeJournal(trades, startBal) {
    btTradesAll = trades.filter(t => t.pnl !== 0);
    btStartBal = startBal || 100;
    btTradeOffset = 0;
    const card = document.getElementById('bt-trades-card');
    if (card) card.style.display = '';
    renderBtTrades();
  }

  function renderBtTrades() {
    const el = document.getElementById('bt-trades');
    const pagEl = document.getElementById('bt-trades-pag');
    if (!el) return;

    const page = btTradesAll.slice(btTradeOffset, btTradeOffset + 50);
    el.innerHTML = `<table class="tbl"><thead><tr>
      <th>Time</th><th>Side</th><th>Price</th><th>Size</th><th>PnL %</th><th>Balance %</th>
    </tr></thead><tbody>
      ${page.map(t => {
        const pnlPct = (t.pnl / btStartBal * 100).toFixed(2);
        const balPct = (t.balance_after / btStartBal * 100).toFixed(1);
        return `<tr>
        <td class="mono" style="font-size:11px;">${TB.utils.fmtDateShort(t.time_ms)}</td>
        <td><span class="badge ${t.side === 'buy' ? 'badge-green' : 'badge-red'}">${TB.utils.esc((t.side || '').toUpperCase())}</span></td>
        <td class="mono">${TB.utils.fmtPrice(t.price)}</td>
        <td class="mono">${TB.utils.esc(String(t.size))}</td>
        <td class="mono ${TB.utils.pnlClass(t.pnl)}">${pnlPct}%</td>
        <td class="mono">${balPct}%</td>
      </tr>`;
      }).join('')}
    </tbody></table>`;

    if (pagEl && btTradesAll.length > 50) {
      const total = Math.ceil(btTradesAll.length / 50);
      const curr = Math.floor(btTradeOffset / 50) + 1;
      pagEl.innerHTML = `
        <button class="btn btn-sm" data-action="bt-prev" ${btTradeOffset === 0 ? 'disabled' : ''}>Prev</button>
        <span class="page-info">${curr} / ${total}</span>
        <button class="btn btn-sm" data-action="bt-next" ${btTradeOffset + 50 >= btTradesAll.length ? 'disabled' : ''}>Next</button>
      `;
      pagEl.querySelector('[data-action="bt-prev"]').addEventListener('click', () => {
        btTradeOffset = Math.max(0, btTradeOffset - 50); renderBtTrades();
      });
      pagEl.querySelector('[data-action="bt-next"]').addEventListener('click', () => {
        btTradeOffset += 50; renderBtTrades();
      });
    }
  }

  function initEquityChart(results) {
    const el = document.getElementById('bt-equity');
    if (!el || typeof LightweightCharts === 'undefined') return;
    if (equityChart) { equityChart.remove(); }
    equitySeries = [];

    equityChart = LightweightCharts.createChart(el, {
      width: el.clientWidth,
      height: 300,
      layout: { background: { color: '#161b22' }, textColor: '#8b949e' },
      grid: { vertLines: { color: '#21262d' }, horzLines: { color: '#21262d' } },
      timeScale: { timeVisible: true, borderColor: '#30363d' },
      rightPriceScale: { borderColor: '#30363d' },
      crosshair: { mode: 0 },
    });

    const colors = ['#58a6ff', '#3fb950', '#f0883e', '#bc8cff', '#f85149'];
    const coins = Object.keys(results);

    coins.forEach((coin, i) => {
      const r = results[coin];
      if (!r.equity_curve || r.equity_curve.length === 0) return;

      const startBal = r.start_balance || 100;

      const series = equityChart.addLineSeries({
        color: colors[i % colors.length],
        lineWidth: 2,
        title: coin,
      });

      // Convert equity to % of initial balance
      const points = r.equity_curve.map(d => ({
        time: Math.floor(d.time_ms / 1000),
        value: (d.equity / startBal) * 100,
      }));
      series.setData(points);
      equitySeries.push(series);
    });

    const ro = new ResizeObserver(() => {
      if (equityChart && el.clientWidth > 0) equityChart.resize(el.clientWidth, 300);
    });
    ro.observe(el);
  }

  async function loadHistory() {
    const data = await TB.api.get('/api/backtest/history?limit=500');
    const el = document.getElementById('bt-history');
    if (!el) return;

    if (!data || data.length === 0) {
      el.innerHTML = '<div class="c-text2 text-center" style="padding:20px;">No history</div>';
      return;
    }

    el.innerHTML = `<table class="tbl"><thead><tr>
      <th>Date</th><th>Strategy</th><th>Coin</th><th>Return</th><th>Sharpe</th><th>DD</th><th>WR</th><th>Trades</th><th>PF</th><th>Verdict</th>
    </tr></thead><tbody>
      ${data.map(r => `<tr>
        <td class="mono" style="font-size:11px;">${TB.utils.fmtDateShort(r.timestamp_ms)}</td>
        <td>${TB.utils.esc(r.strategy)}</td>
        <td class="mono">${TB.utils.esc(r.coin)}</td>
        <td class="mono ${TB.utils.pnlClass(r.return_pct)}">${TB.utils.fmtPct(r.return_pct)}</td>
        <td class="mono">${r.sharpe != null ? r.sharpe.toFixed(2) : '\u2014'}</td>
        <td class="mono c-red">${r.max_dd != null ? r.max_dd.toFixed(1) + '%' : '\u2014'}</td>
        <td class="mono">${r.win_rate != null ? r.win_rate.toFixed(0) + '%' : '\u2014'}</td>
        <td class="mono">${r.total_trades || 0}</td>
        <td class="mono">${r.profit_factor != null ? (r.profit_factor >= 999 ? '\u221e' : r.profit_factor.toFixed(2)) : '\u2014'}</td>
        <td><span class="badge verdict-${r.verdict || ''}">${r.verdict || '\u2014'}</span></td>
      </tr>`).join('')}
    </tbody></table>`;
  }

  async function clearHistory() {
    if (!confirm('Clear all backtest history?')) return;
    const result = await TB.api.del('/api/backtest/history');
    if (result) {
      TB.toast.show('success', `Deleted ${result.deleted} entries`);
      loadHistory();
    }
  }

  TB.pages.backtest = { mount, unmount };
})();
