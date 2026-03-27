(function() {
  'use strict';
  window.TB = window.TB || {};
  TB.pages = TB.pages || {};

  let intervals = [];
  let prevPrices = {};
  let midsListener = null;
  let trackedCoins = []; // loaded from strategies API
  let volumes = {}; // 24h notional volumes from asset contexts

  function mount(container) {
    container.innerHTML = `
      <div class="section-header">
        <h2>Market</h2>
        <span id="mkt-live" style="display:flex;align-items:center;gap:6px;font-size:11px;color:var(--text2);">
          <span class="live-dot pulse"></span> LIVE
        </span>
      </div>

      <div class="card mb-16">
        <div class="card-header"><span class="live-dot pulse"></span> Live Prices</div>
        <div class="card-body">
          <div id="price-row" class="price-grid"></div>
        </div>
      </div>

      <div class="grid-2 mb-16 market-mid-grid" style="grid-template-columns:1fr 1fr;">
        <div class="card">
          <div class="card-header">Market Sentiment</div>
          <div class="card-body" id="mkt-sentiment"></div>
        </div>
        <div class="card">
          <div class="card-header">Global Market</div>
          <div class="card-body" id="mkt-global-forex"></div>
        </div>
      </div>

      <div class="card">
        <div class="card-header">AI Daily Digest</div>
        <div class="card-body" id="mkt-digest"></div>
      </div>
    `;

    // Load tracked coins and volumes, then render prices
    loadTrackedCoins();
    loadVolumes();

    midsListener = renderPrices;
    TB.state.on('mids', midsListener);
    renderPrices(TB.state.get('mids'));

    loadOverview();
    loadGlobalMarket();
    loadDigest();

    intervals.push(setInterval(loadOverview, 120000));
    intervals.push(setInterval(loadGlobalMarket, 120000));
    intervals.push(setInterval(loadVolumes, 120000));
    // Check once per minute if it's noon and digest needs refresh
    intervals.push(setInterval(checkDigestRefresh, 60000));
  }

  function unmount() {
    intervals.forEach(clearInterval);
    intervals = [];
    if (midsListener) { TB.state.off('mids', midsListener); midsListener = null; }
  }

  // --- Load tracked coins from strategies API ---
  async function loadTrackedCoins() {
    const strats = await TB.api.get('/api/strategies');
    if (!strats) return;
    const coinSet = new Set();
    strats.forEach(s => {
      if (s.coins && s.status !== 'DISABLED') s.coins.forEach(c => coinSet.add(c));
    });
    trackedCoins = [...coinSet];
    renderPrices(TB.state.get('mids'));
  }

  // --- Load 24h volumes ---
  async function loadVolumes() {
    const data = await TB.api.get('/api/market/volumes');
    if (data) {
      volumes = data;
      renderPrices(TB.state.get('mids'));
    }
  }

  // --- Live Prices (sorted by volume) ---
  function renderPrices(mids) {
    const el = document.getElementById('price-row');
    if (!el) return;

    if (!mids || Object.keys(mids).length === 0) {
      el.innerHTML = '<div class="c-text2" style="padding:12px 0;">Start the bot to see live prices</div>';
      return;
    }

    const tracked = [];
    const other = [];

    Object.keys(mids).forEach(coin => {
      if (trackedCoins.includes(coin)) tracked.push(coin);
      else other.push(coin);
    });

    // Sort both groups by 24h volume descending
    const byVol = (a, b) => (volumes[b] || 0) - (volumes[a] || 0);
    tracked.sort(byVol);
    other.sort(byVol);

    const allCoins = [...tracked, ...other];

    el.innerHTML = allCoins.map(coin => {
      const price = mids[coin];
      const prev = prevPrices[coin];
      let flashClass = '';
      if (prev !== undefined && price !== prev) {
        flashClass = price > prev ? 'flash-up' : 'flash-down';
      }
      const isTracked = trackedCoins.includes(coin);
      const vol = volumes[coin];
      const volStr = vol ? '$' + fmtVolume(vol) : '';
      return `<div class="price-cell${isTracked ? ' tracked' : ''}">
        <div class="coin-name">${TB.utils.esc(coin)}</div>
        <div class="coin-price ${flashClass}">${TB.utils.fmtPrice(price, coin)}</div>
        ${volStr ? `<div class="coin-vol">${volStr}</div>` : ''}
      </div>`;
    }).join('');

    prevPrices = { ...mids };
  }

  function fmtVolume(v) {
    if (v >= 1e9) return (v / 1e9).toFixed(1) + 'B';
    if (v >= 1e6) return (v / 1e6).toFixed(1) + 'M';
    if (v >= 1e3) return (v / 1e3).toFixed(0) + 'K';
    return v.toFixed(0);
  }

  // --- Market Sentiment (F&G gauge + Phase) ---
  async function loadOverview() {
    const data = await TB.api.get('/api/market/overview');
    const el = document.getElementById('mkt-sentiment');
    if (!el || !data) return;

    const fg = data.fear_greed || 50;
    const fgLabel = data.fear_greed_label || 'N/A';
    const phase = data.phase || 'unknown';
    const strategies = data.recommended_strategies || [];
    const ind = data.indicators || {};

    const fgColor = getFGColor(fg);

    const phaseLabels = { bull: 'Bullish', bear: 'Bearish', range: 'Ranging', high_vol: 'High Volatility', unknown: 'Unknown' };

    // If phase is unknown and no indicators, show a "waiting for data" message
    const hasData = phase !== 'unknown' || (ind.sma20 && ind.sma20 > 0);

    el.innerHTML = `
      <div class="market-sentiment">
        <div>
          ${buildFGGauge(fg, fgColor)}
          <div class="fg-gauge-label">${TB.utils.esc(fgLabel)}</div>
        </div>
        <div>
          <div style="margin-bottom:10px;">
            <span class="phase-badge ${TB.utils.esc(phase)}">${TB.utils.esc(phaseLabels[phase] || phase.toUpperCase())}</span>
          </div>
          ${!hasData ? `
            <div style="font-size:11px;color:var(--text3);margin-top:4px;">Waiting for BTC candle data...</div>
          ` : ''}
          ${ind.sma20 ? `
          <div class="phase-indicators">
            <div class="phase-indicator">
              <div class="ind-label">SMA 20</div>
              <div class="ind-value">$${TB.utils.fmtPrice(ind.sma20)}</div>
            </div>
            <div class="phase-indicator">
              <div class="ind-label">SMA 50</div>
              <div class="ind-value">$${TB.utils.fmtPrice(ind.sma50)}</div>
            </div>
            <div class="phase-indicator">
              <div class="ind-label">ATR</div>
              <div class="ind-value">${ind.atr_pct}%</div>
            </div>
            ${ind.price ? `<div class="phase-indicator">
              <div class="ind-label">BTC Price</div>
              <div class="ind-value">$${TB.utils.fmtPrice(ind.price)}</div>
            </div>` : ''}
          </div>` : ''}
          ${strategies.length ? `
          <div class="strategy-pills">
            ${strategies.map(s => `<span class="strategy-pill">${TB.utils.esc(s.replace(/_/g, ' '))}</span>`).join('')}
          </div>` : ''}
        </div>
      </div>
    `;
  }

  function buildFGGauge(value, color) {
    const angle = (value / 100) * 180;
    const startAngle = 180;
    const endAngle = startAngle + angle;

    const bgArc = describeArc(80, 80, 65, 180, 360);
    const valArc = describeArc(80, 80, 65, startAngle, endAngle);

    const gradStops = [
      { offset: '0%', color: '#f85149' },
      { offset: '25%', color: '#f0883e' },
      { offset: '50%', color: '#d29922' },
      { offset: '75%', color: '#3fb950' },
      { offset: '100%', color: '#3fb950' },
    ];

    return `
      <div class="fg-gauge">
        <svg viewBox="0 0 160 95" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <linearGradient id="fg-grad" x1="0%" y1="0%" x2="100%" y2="0%">
              ${gradStops.map(s => `<stop offset="${s.offset}" stop-color="${s.color}" />`).join('')}
            </linearGradient>
          </defs>
          <path d="${bgArc}" fill="none" stroke="url(#fg-grad)" stroke-width="12" stroke-linecap="round" opacity="0.2" />
          <path d="${valArc}" fill="none" stroke="url(#fg-grad)" stroke-width="12" stroke-linecap="round" />
          ${buildNeedle(80, 80, 55, value)}
        </svg>
        <div class="fg-gauge-value" style="color:${color};">${value}</div>
      </div>
    `;
  }

  function buildNeedle(cx, cy, r, value) {
    const angle = 180 + (value / 100) * 180;
    const rad = (angle * Math.PI) / 180;
    const x = cx + r * Math.cos(rad);
    const y = cy + r * Math.sin(rad);
    return `<circle cx="${cx}" cy="${cy}" r="4" fill="var(--text2)" />
            <line x1="${cx}" y1="${cy}" x2="${x}" y2="${y}" stroke="var(--text)" stroke-width="2" stroke-linecap="round" />`;
  }

  function describeArc(cx, cy, r, startAngle, endAngle) {
    const start = polarToCartesian(cx, cy, r, endAngle);
    const end = polarToCartesian(cx, cy, r, startAngle);
    const largeArc = endAngle - startAngle <= 180 ? '0' : '1';
    return `M ${start.x} ${start.y} A ${r} ${r} 0 ${largeArc} 0 ${end.x} ${end.y}`;
  }

  function polarToCartesian(cx, cy, r, angleDeg) {
    const rad = ((angleDeg) * Math.PI) / 180;
    return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
  }

  function getFGColor(v) {
    if (v < 25) return '#f85149';
    if (v < 45) return '#f0883e';
    if (v < 55) return '#d29922';
    if (v < 75) return '#3fb950';
    return '#3fb950';
  }

  // --- Global Market ---
  async function loadGlobalMarket() {
    const globalData = await TB.api.get('/api/market/global');
    const el = document.getElementById('mkt-global-forex');
    if (!el) return;

    const g = globalData || {};

    el.innerHTML = `
      <div class="global-forex-grid">
        <div class="gf-metric">
          <div class="gf-label">BTC Dominance</div>
          <div class="gf-value">${g.btc_dominance || 0}%</div>
        </div>
        <div class="gf-metric">
          <div class="gf-label">Total Market Cap</div>
          <div class="gf-value">$${TB.utils.fmtBigNum(g.total_market_cap || 0)}</div>
        </div>
        <div class="gf-metric">
          <div class="gf-label">TOTAL2 (ex-BTC)</div>
          <div class="gf-value">$${TB.utils.fmtBigNum(g.total2 || 0)}</div>
        </div>
        <div class="gf-metric">
          <div class="gf-label">TOTAL3 (ex-BTC-ETH)</div>
          <div class="gf-value">$${TB.utils.fmtBigNum(g.total3 || 0)}</div>
        </div>
      </div>
    `;
  }

  // --- AI Daily Digest ---
  async function loadDigest() {
    const el = document.getElementById('mkt-digest');
    if (!el) return;
    el.innerHTML = '<div class="c-text2" style="padding:12px 0;">Loading digest...</div>';

    const data = await TB.api.get('/api/market/digest');
    renderDigest(el, data);
  }

  let lastDigestRefreshDate = '';
  function checkDigestRefresh() {
    const now = new Date();
    const today = now.toISOString().slice(0, 10);
    const hour = now.getHours();
    // Auto-refresh at noon if not already refreshed today
    if (hour >= 12 && lastDigestRefreshDate !== today) {
      lastDigestRefreshDate = today;
      loadDigest();
    }
  }

  function renderDigest(el, data) {
    if (!data) {
      el.innerHTML = '<div class="digest-error"><span class="error-icon">!</span>Failed to load digest</div>';
      return;
    }

    if (data.error) {
      el.innerHTML = `<div class="digest-error">
        <span class="error-icon">&#9888;</span>
        <div>${TB.utils.esc(data.error)}</div>
      </div>`;
      return;
    }

    const sentiment = data.sentiment || 'unknown';
    const reason = data.sentiment_reason || '';
    const genAt = data.generated_at ? timeAgo(data.generated_at) : '';
    const sources = data.sources || [];

    el.innerHTML = `
      <div class="digest-header">
        <span class="digest-sentiment-badge ${TB.utils.esc(sentiment)}">${TB.utils.esc(sentiment)}</span>
        ${reason ? `<span style="font-size:12px;color:var(--text2);">${TB.utils.esc(reason)}</span>` : ''}
        ${data.stale ? '<span class="badge badge-yellow">STALE</span>' : ''}
        ${data.cached ? '<span class="badge badge-gray">CACHED</span>' : ''}
        ${genAt ? `<span class="digest-time">${TB.utils.esc(genAt)}</span>` : ''}
      </div>
      ${renderDigestSection('Points Cles', data.points, 'kp')}
      ${renderDigestSection('Events', data.events, 'ev')}
      ${renderDigestSection('Trends', data.trends, 'tr')}
      ${sources.length ? `<div style="margin-top:10px;font-size:10px;color:var(--text3);">Sources: ${sources.map(s => TB.utils.esc(s)).join(' / ')}${data.article_count ? ` (${data.article_count} articles)` : ''}</div>` : ''}
    `;
  }

  function renderDigestSection(title, items, cls) {
    if (!items || items.length === 0) return '';
    return `
      <div class="digest-section">
        <div class="digest-section-title">${TB.utils.esc(title)}</div>
        <div class="digest-items">
          ${items.map(item => `<div class="digest-item ${cls}">${TB.utils.esc(item)}</div>`).join('')}
        </div>
      </div>
    `;
  }

  function timeAgo(val) {
    try {
      // Handle both ISO string and unix timestamp (seconds or ms)
      let ts;
      if (typeof val === 'number') {
        ts = val > 1e12 ? val : val * 1000;
      } else {
        ts = new Date(val).getTime();
      }
      const diff = Math.floor((Date.now() - ts) / 1000);
      if (diff < 0) return 'just now';
      if (diff < 60) return 'just now';
      if (diff < 3600) return Math.floor(diff / 60) + 'm ago';
      if (diff < 86400) return Math.floor(diff / 3600) + 'h ago';
      return Math.floor(diff / 86400) + 'd ago';
    } catch { return ''; }
  }

  TB.pages.market = { mount, unmount };
})();
