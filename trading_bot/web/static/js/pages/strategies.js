(function() {
  'use strict';
  window.TB = window.TB || {};
  TB.pages = TB.pages || {};

  let intervals = [];
  let selectedStrategy = null;

  function mount(container) {
    container.innerHTML = `
      <div class="section-header"><h2>Strategies</h2></div>
      <div class="grid-2">
        <div class="card">
          <div class="card-header">Strategy List</div>
          <div class="card-body overflow-auto" id="strat-list" style="max-height:600px;"></div>
        </div>
        <div class="card">
          <div class="card-header" id="code-header">Code Viewer</div>
          <div class="card-body" id="code-viewer">
            <div class="c-text2 text-center" style="padding:40px;">Select a strategy to view code</div>
          </div>
        </div>
      </div>
    `;

    loadStrategies();
    intervals.push(setInterval(loadStrategies, 10000));
  }

  function unmount() {
    intervals.forEach(clearInterval);
    intervals = [];
    selectedStrategy = null;
  }

  async function loadStrategies() {
    const data = await TB.api.get('/api/strategies');
    const el = document.getElementById('strat-list');
    if (!el || !data) return;

    if (data.length === 0) {
      el.innerHTML = '<div class="c-text2 text-center" style="padding:20px;">No strategies loaded</div>';
      return;
    }

    el.innerHTML = data.map(s => {
      const roleColors = { primary: 'badge-blue', secondary: 'badge-purple', '': 'badge-gray' };
      const roleBadge = s.role ? `<span class="badge ${roleColors[s.role] || 'badge-gray'}">${TB.utils.esc(s.role.toUpperCase())}</span>` : '';
      const modeBadge = s.paper_mode ? '<span class="badge badge-yellow">PAPER</span>' : '<span class="badge badge-green">LIVE</span>';
      const statusBadge = s.status === 'ERRORED' ? '<span class="badge badge-red">ERRORED</span>' : s.status === 'DISABLED' ? '<span class="badge badge-gray">DISABLED</span>' : s.status === 'STOPPED' ? '<span class="badge badge-yellow">STOPPED</span>' : '';
      const wrWarn = s.win_rate !== null && s.win_rate < 30 ? ' c-red' : '';

      return `<div style="padding:12px;border-bottom:1px solid var(--border);cursor:pointer;" onclick="TB._selectStrategy('${TB.utils.esc(s.name)}')">
        <div class="flex gap-8 mb-12" style="align-items:center;flex-wrap:wrap;">
          <strong>${TB.utils.esc(s.name)}</strong>
          ${roleBadge} ${modeBadge} ${statusBadge}
        </div>
        <div style="font-size:11px;color:var(--text2);margin-bottom:6px;">Coins: ${s.coins.map(c => `<span class="badge badge-gray">${TB.utils.esc(c)}</span>`).join(' ')}</div>
        <div class="flex gap-16" style="font-size:12px;">
          <span>Trades: <strong>${s.trades}</strong></span>
          <span class="${wrWarn}">WR: <strong>${s.win_rate !== null ? s.win_rate + '%' : '—'}</strong></span>
          <span class="${TB.utils.pnlClass(s.pnl)}">PnL: <strong>${TB.utils.fmtPnl(s.pnl)}</strong></span>
          ${s.position ? `<span>Pos: <span class="badge ${s.position === 'BUY' ? 'badge-green' : 'badge-red'}">${s.position}</span></span>` : ''}
        </div>
        ${s.win_rate_per_coin && Object.keys(s.win_rate_per_coin).length > 0 ?
          `<div style="font-size:10px;color:var(--text3);margin-top:4px;">WR/coin: ${Object.entries(s.win_rate_per_coin).map(([c,w]) => `${c}:${w}%`).join(' ')}</div>` : ''}
        <div class="flex gap-8 mt-8">
          <button class="btn btn-sm" onclick="event.stopPropagation();TB._toggleStrategy('${TB.utils.esc(s.name)}')">
            ${s.status === 'DISABLED' || s.status === 'ERRORED' ? 'Enable' : 'Disable'}
          </button>
        </div>
      </div>`;
    }).join('');
  }

  TB._selectStrategy = async (name) => {
    selectedStrategy = name;
    const header = document.getElementById('code-header');
    const viewer = document.getElementById('code-viewer');
    if (header) header.textContent = 'Code: ' + name;
    if (viewer) viewer.innerHTML = '<div class="c-text2">Loading...</div>';

    const data = await TB.api.get('/api/strategies/' + encodeURIComponent(name) + '/code');
    if (!viewer || !data || data.error) {
      if (viewer) viewer.innerHTML = '<div class="c-red">Failed to load</div>';
      return;
    }

    const escaped = TB.utils.esc(data.code);
    viewer.innerHTML = `<div style="font-size:11px;color:var(--text2);margin-bottom:8px;">${TB.utils.esc(data.file)} — ${data.lines} lines</div>
      <pre class="language-python line-numbers"><code class="language-python">${escaped}</code></pre>`;

    if (typeof Prism !== 'undefined') {
      Prism.highlightAllUnder(viewer);
    }
  };

  TB._toggleStrategy = async (name) => {
    const result = await TB.api.post('/api/strategies/' + encodeURIComponent(name) + '/toggle');
    if (result) {
      TB.toast.show('success', `Strategy ${name}: ${result.status}`);
      loadStrategies();
    }
  };

  TB.pages.strategies = { mount, unmount };
})();
