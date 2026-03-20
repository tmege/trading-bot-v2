(function() {
  'use strict';
  window.TB = window.TB || {};
  TB.pages = TB.pages || {};

  let intervals = [];
  let logSSE = null;
  let term = null;
  let fitAddon = null;
  let equityChart = null;
  let equitySeries = null;
  let tradeOffset = 0;
  let tradeFilters = { coin: '', strategy: '' };

  function mount(container) {
    container.innerHTML = `
      <div id="dash-alerts"></div>
      <div id="dash-paper-banner"></div>
      <div class="section-header">
        <h2>Dashboard</h2>
        <div class="flex gap-8" id="dash-controls">
          <span id="dash-uptime" class="c-text2" style="font-size:11px;line-height:28px;"></span>
          <button class="btn btn-danger btn-sm" id="btn-stop">Stop Bot</button>
        </div>
      </div>
      <div class="metrics-grid" id="dash-metrics"></div>
      <div class="section-header"><h2>Performance Monitoring</h2></div>
      <div class="metrics-grid" id="dash-perf"></div>
      <div class="card mb-16">
        <div class="card-header">Equity Curve</div>
        <div class="card-body"><div id="equity-chart" class="chart-container"></div></div>
      </div>
      <div class="grid-2 mb-16">
        <div class="card">
          <div class="card-header">Open Positions</div>
          <div class="card-body overflow-auto" style="max-height:320px;" id="dash-positions"></div>
        </div>
        <div class="card">
          <div class="card-header">
            <span>Recent Trades</span>
            <div class="flex gap-8">
              <select id="trade-filter-coin" class="input" style="width:80px;"><option value="">All</option></select>
              <select id="trade-filter-strategy" class="input" style="width:120px;"><option value="">All</option></select>
              <button class="btn btn-sm" id="btn-export-csv">CSV</button>
            </div>
          </div>
          <div class="card-body overflow-auto" style="max-height:320px;" id="dash-trades"></div>
          <div class="pagination" id="dash-trades-pag"></div>
        </div>
      </div>
      <div class="card">
        <div class="card-header">Terminal</div>
        <div class="card-body" style="padding:0;">
          <div id="terminal" class="terminal-container"></div>
        </div>
      </div>
    `;

    document.getElementById('btn-stop').onclick = async () => {
      const btn = document.getElementById('btn-stop');
      const isRunning = btn.dataset.running === 'true';
      if (isRunning) {
        if (confirm('Stop the bot?')) {
          btn.disabled = true;
          await TB.api.post('/api/bot/stop');
          TB.toast.show('warning', 'Bot stopping...');
          btn.disabled = false;
        }
      } else {
        btn.disabled = true;
        const res = await TB.api.post('/api/bot/start');
        if (res && res.status === 'started') {
          TB.toast.show('success', 'Bot started');
        } else {
          TB.toast.show('error', res ? (res.error || 'Start failed') : 'Start failed');
        }
        btn.disabled = false;
      }
    };

    document.getElementById('btn-export-csv').onclick = () => {
      const q = new URLSearchParams(tradeFilters).toString();
      window.open('/api/trades/export?' + q, '_blank');
    };

    document.getElementById('trade-filter-coin').onchange = (e) => {
      tradeFilters.coin = e.target.value; tradeOffset = 0; loadTrades();
    };
    document.getElementById('trade-filter-strategy').onchange = (e) => {
      tradeFilters.strategy = e.target.value; tradeOffset = 0; loadTrades();
    };

    loadAccount();
    loadPerformance();
    loadPositions();
    loadTrades();
    initEquityChart();
    loadEquityCurve();
    initTerminal();

    intervals.push(setInterval(loadAccount, 3000));
    intervals.push(setInterval(loadPositions, 3000));
    intervals.push(setInterval(loadTrades, 10000));
    intervals.push(setInterval(loadPerformance, 30000));
    intervals.push(setInterval(loadEquityCurve, 60000));
  }

  function unmount() {
    intervals.forEach(clearInterval);
    intervals = [];
    if (logSSE) { logSSE.close(); logSSE = null; }
    if (term) { term.dispose(); term = null; }
    if (equityChart) { equityChart.remove(); equityChart = null; equitySeries = null; }
  }

  async function loadAccount() {
    const [acc, status] = await Promise.all([
      TB.api.get('/api/account'),
      TB.api.get('/api/bot/status'),
    ]);
    if (!acc) return;

    const el = document.getElementById('dash-metrics');
    if (!el) return;

    el.innerHTML = `
      ${metricCard('Balance', '$' + TB.utils.fmtNum(acc.account_value, 2), '')}
      ${metricCard('Cumul PnL', TB.utils.fmtPnl(acc.cumulative_pnl), '', acc.cumulative_pnl)}
      ${metricCard('Daily PnL', TB.utils.fmtPnl(acc.daily_pnl) + ' <span style="font-size:12px;color:var(--text2)">(' + TB.utils.fmtPnl(acc.daily_unrealized_pnl) + ' unreal)</span>', '', acc.daily_pnl)}
      ${metricCard('Daily Fees', '$' + TB.utils.fmtNum(acc.daily_fees, 4), '')}
      ${metricCard('Trades Today', String(acc.daily_trades), '')}
      ${metricCard('Positions', String(acc.open_positions), '')}
    `;

    if (status) {
      const up = document.getElementById('dash-uptime');
      if (up) up.textContent = 'Uptime: ' + TB.utils.fmtDuration(status.uptime);

      const banner = document.getElementById('dash-paper-banner');
      if (banner) {
        banner.innerHTML = status.paper_trading
          ? '<div class="alerts-banner yellow" style="text-align:center;font-weight:600;">PAPER TRADING MODE</div>'
          : '';
      }

      const btn = document.getElementById('btn-stop');
      if (btn) {
        btn.dataset.running = String(!!status.running);
        if (status.running) {
          btn.textContent = 'Stop Bot';
          btn.className = 'btn btn-danger btn-sm';
        } else {
          btn.textContent = 'Start Bot';
          btn.className = 'btn btn-primary btn-sm';
        }
      }
    }
  }

  async function loadPerformance() {
    const data = await TB.api.get('/api/account/performance');
    if (!data) return;

    const el = document.getElementById('dash-perf');
    if (!el) return;

    const alertClass = (metric) => {
      const a = (data.alerts || []).find(x => x.metric === metric);
      if (!a) return 'alert-green';
      return a.level === 'red' ? 'alert-red' : 'alert-yellow';
    };

    el.innerHTML = `
      ${metricCard('Sharpe 30d', String(data.sharpe_30d), '', null, alertClass('sharpe'))}
      ${metricCard('Max Drawdown', data.max_drawdown_pct + '%', '', null, alertClass('max_dd'))}
      ${metricCard('Fee Drag', data.fee_drag_pct + '%', '', null, alertClass('fee_drag'))}
      ${metricCard('Win Rate 7d', data.win_rate_7d + '%', '', null, alertClass('strategy_wr'))}
      ${metricCard('Trades/Day', String(data.trades_per_day), '', null, alertClass('trades_per_day'))}
      ${metricCard('PnL 7d', TB.utils.fmtPnl(data.pnl_7d), '', data.pnl_7d, alertClass('pnl_7d'))}
    `;

    const alertsEl = document.getElementById('dash-alerts');
    if (alertsEl && data.alerts && data.alerts.length > 0) {
      const hasRed = data.alerts.some(a => a.level === 'red');
      alertsEl.innerHTML = `<div class="alerts-banner ${hasRed ? '' : 'yellow'}">
        ${data.alerts.map(a => `<div>${TB.utils.esc(a.message)}</div>`).join('')}
      </div>`;
    } else if (alertsEl) {
      alertsEl.innerHTML = '';
    }
  }

  async function loadPositions() {
    const data = await TB.api.get('/api/positions');
    const el = document.getElementById('dash-positions');
    if (!el) return;

    if (!data || data.length === 0) {
      el.innerHTML = '<div class="c-text2 text-center" style="padding:20px;">No open positions</div>';
      return;
    }

    el.innerHTML = `<table class="tbl"><thead><tr>
      <th>Coin</th><th>Side</th><th>Size</th><th>Entry</th><th>uPnL</th><th>ROI%</th><th>Lev</th>
    </tr></thead><tbody>
      ${data.map(p => `<tr>
        <td class="mono">${TB.utils.esc(p.coin)}</td>
        <td><span class="badge ${p.side === 'LONG' ? 'badge-green' : 'badge-red'}">${p.side}</span></td>
        <td class="mono">${TB.utils.esc(String(p.size))}</td>
        <td class="mono">${TB.utils.fmtPrice(p.entry_px)}</td>
        <td class="mono ${TB.utils.pnlClass(p.unrealized_pnl)}">${TB.utils.fmtPnl(p.unrealized_pnl)}</td>
        <td class="mono ${TB.utils.pnlClass(p.roi_pct)}">${TB.utils.fmtPct(p.roi_pct)}</td>
        <td class="mono">${p.leverage}x</td>
      </tr>`).join('')}
    </tbody></table>`;
  }

  async function loadTrades() {
    const params = new URLSearchParams({ limit: 100, offset: tradeOffset, ...tradeFilters });
    const data = await TB.api.get('/api/trades?' + params);
    const el = document.getElementById('dash-trades');
    if (!el || !data) return;

    const trades = data.trades || [];
    if (trades.length === 0) {
      el.innerHTML = '<div class="c-text2 text-center" style="padding:20px;">No trades</div>';
      return;
    }

    el.innerHTML = `<table class="tbl"><thead><tr>
      <th>Time</th><th>Coin</th><th>Side</th><th>Price</th><th>Size</th><th>PnL</th><th>Fee</th>
    </tr></thead><tbody>
      ${trades.map(t => `<tr>
        <td class="mono" style="font-size:11px;">${TB.utils.fmtDateShort(t.time_ms)}</td>
        <td class="mono">${TB.utils.esc(t.coin)}</td>
        <td><span class="badge ${t.side === 'buy' ? 'badge-green' : 'badge-red'}">${TB.utils.esc((t.side || '').toUpperCase())}</span></td>
        <td class="mono">${TB.utils.fmtPrice(t.price)}</td>
        <td class="mono">${TB.utils.esc(String(t.size))}</td>
        <td class="mono ${TB.utils.pnlClass(t.closed_pnl)}">${TB.utils.fmtPnl(t.closed_pnl)}</td>
        <td class="mono c-text2">${TB.utils.fmtNum(t.fee, 4)}</td>
      </tr>`).join('')}
    </tbody></table>`;

    const pagEl = document.getElementById('dash-trades-pag');
    if (pagEl && data.total > 100) {
      const totalPages = Math.ceil(data.total / 100);
      const currentPage = Math.floor(tradeOffset / 100) + 1;
      pagEl.innerHTML = `
        <button class="btn btn-sm" data-action="trade-prev" ${tradeOffset === 0 ? 'disabled' : ''}>Prev</button>
        <span class="page-info">${currentPage} / ${totalPages}</span>
        <button class="btn btn-sm" data-action="trade-next" ${tradeOffset + 100 >= data.total ? 'disabled' : ''}>Next</button>
      `;
      pagEl.querySelector('[data-action="trade-prev"]').addEventListener('click', () => {
        tradeOffset = Math.max(0, tradeOffset - 100); loadTrades();
      });
      pagEl.querySelector('[data-action="trade-next"]').addEventListener('click', () => {
        tradeOffset += 100; loadTrades();
      });
    }
  }

  function initEquityChart() {
    const el = document.getElementById('equity-chart');
    if (!el || typeof LightweightCharts === 'undefined') return;

    equityChart = LightweightCharts.createChart(el, {
      width: el.clientWidth,
      height: 180,
      layout: { background: { color: '#161b22' }, textColor: '#8b949e' },
      grid: { vertLines: { color: '#21262d' }, horzLines: { color: '#21262d' } },
      timeScale: { timeVisible: true, borderColor: '#30363d' },
      rightPriceScale: { borderColor: '#30363d' },
      crosshair: { mode: 0 },
    });
    equitySeries = equityChart.addAreaSeries({
      topColor: 'rgba(88,166,255,0.3)',
      bottomColor: 'rgba(88,166,255,0.02)',
      lineColor: '#58a6ff',
      lineWidth: 2,
    });

    const ro = new ResizeObserver(() => {
      if (equityChart && el.clientWidth > 0) equityChart.resize(el.clientWidth, 180);
    });
    ro.observe(el);
  }

  async function loadEquityCurve() {
    if (!equitySeries) return;
    const data = await TB.api.get('/api/account/equity-curve?days=30');
    if (!data || data.length === 0) return;

    const points = data.map(d => ({
      time: Math.floor(d.time_ms / 1000),
      value: d.equity,
    }));
    equitySeries.setData(points);
  }

  function initTerminal() {
    const el = document.getElementById('terminal');
    if (!el || typeof Terminal === 'undefined') return;

    term = new Terminal({
      theme: {
        background: '#0d1117',
        foreground: '#e6edf3',
        cursor: '#58a6ff',
      },
      fontSize: 12,
      fontFamily: "'SF Mono', 'Fira Code', monospace",
      scrollback: 5000,
      disableStdin: true,
      convertEol: true,
    });

    if (typeof FitAddon !== 'undefined') {
      fitAddon = new FitAddon.FitAddon();
      term.loadAddon(fitAddon);
    }

    term.open(el);
    if (fitAddon) fitAddon.fit();

    const ro = new ResizeObserver(() => { if (fitAddon) fitAddon.fit(); });
    ro.observe(el);

    logSSE = new EventSource('/api/logs/stream?api_key=' + encodeURIComponent(window.__TB_API_KEY__ || ''));
    logSSE.onmessage = (e) => {
      if (e.data && term) {
        term.writeln(e.data.replace(/\\n/g, '\n'));
      }
    };

    // Load initial logs
    TB.api.get('/api/logs?n=100').then(lines => {
      if (lines && Array.isArray(lines)) {
        lines.forEach(l => term.writeln(l));
      }
    });
  }

  function metricCard(label, value, sub, pnlVal, extraClass) {
    const colorClass = pnlVal != null ? TB.utils.pnlClass(pnlVal) : '';
    return `<div class="metric-card ${extraClass || ''}">
      <div class="label">${TB.utils.esc(label)}</div>
      <div class="value ${colorClass}">${value}</div>
      ${sub ? `<div class="sub">${sub}</div>` : ''}
    </div>`;
  }

  TB.pages.dashboard = { mount, unmount };
})();
