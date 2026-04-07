(function() {
  'use strict';
  window.TB = window.TB || {};
  TB.pages = TB.pages || {};

  let intervals = [];
  let selectedStrategy = null;

  // Hardcoded backtest stats (validated 2026-03-23)
  const GROUP_META = {
    '4-strat-mix': {
      label: '4-Strat Mix',
      icon: '\u{1F4CA}',
      stats: [
        '3Y Return: +244%  |  MaxDD: 8.7%',
        'Avg/month: +6.8%  |  Sharpe: ~1.8',
        'Semestres +: 6/6  |  Trades: 1409',
        'Ratio R/DD: 15.4x |  Best: BNB +369%',
      ],
    },
    '4-coin-uniform': {
      label: '4-Coin Uniform Breakout',
      icon: '\u{1F4C8}',
      stats: [
        '3Y Return: +291%  |  MaxDD: 8.4%',
        'Avg/month: +8.1%  |  Sharpe: ~2.36',
        'Trades: ~3 538    |  Best: DOGE +509%',
        'Coins: SOL+BNB+XRP+DOGE  |  Lev: 5x',
        'Profil: SL 0.3% / TP 4% / lb 32 / eq 25%',
      ],
    },
    'high-leverage': {
      label: 'High Leverage',
      icon: '\u26A1',
      stats: [
        '3Y Fixe: BTC +679% ETH +909% |  Sharpe: 3.0-3.4',
        'PF: 1.64-1.67  |  MaxDD: 14-17%  |  Sem+: 5/5',
        'Trades: ~4 692  |  ~4.3/jour  |  WR: ~13.5%',
        'Coins: BTC+ETH  |  Lev: x20',
        'Profil: SL 0.15% / TP 2.5% / lb 6 / eq 15%',
      ],
    },
  };

  function mount(container) {
    container.innerHTML = `
      <div class="section-header"><h2>Strategies</h2></div>
      <div id="group-toggles" style="margin-bottom:16px;"></div>
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

    // Event delegation for group toggles
    document.getElementById('group-toggles').addEventListener('click', function(e) {
      const btn = e.target.closest('[data-action="toggle-group"]');
      if (btn) {
        toggleGroup(btn.dataset.group);
      }
    });

    // Event delegation for strategy list
    document.getElementById('strat-list').addEventListener('click', function(e) {
      const toggleBtn = e.target.closest('[data-action="toggle"]');
      if (toggleBtn) {
        e.stopPropagation();
        toggleStrategy(toggleBtn.dataset.strategy);
        return;
      }
      const row = e.target.closest('[data-action="select"]');
      if (row) {
        selectStrategy(row.dataset.strategy);
      }
    });

    loadStrategies();
    intervals.push(setInterval(loadStrategies, 10000));
  }

  function unmount() {
    intervals.forEach(clearInterval);
    intervals = [];
    selectedStrategy = null;
  }

  function isGroupActive(strategies, groupName) {
    const members = strategies.filter(s => s.group === groupName);
    if (members.length === 0) return false;
    return members.some(s => s.status !== 'DISABLED');
  }

  function renderGroupToggles(strategies) {
    const el = document.getElementById('group-toggles');
    if (!el) return;

    // Collect known groups
    const groups = [];
    const seen = new Set();
    for (const s of strategies) {
      if (s.group && !seen.has(s.group)) {
        seen.add(s.group);
        groups.push(s.group);
      }
    }

    if (groups.length === 0) {
      el.innerHTML = '';
      return;
    }

    el.innerHTML = '<div style="display:flex;gap:16px;flex-wrap:wrap;">' +
      groups.map(g => {
        const meta = GROUP_META[g] || { label: g, stats: [] };
        const active = isGroupActive(strategies, g);
        const borderColor = active ? 'var(--green)' : 'var(--border)';
        const opacity = active ? '1' : '0.5';
        const statusBadge = active
          ? '<span class="badge badge-green">ON</span>'
          : '<span class="badge badge-gray">OFF</span>';

        const isHighLev = g === 'high-leverage';
        const borderStyle = isHighLev && active ? 'border-color:var(--orange,#f59e0b);' : '';
        const iconHtml = meta.icon ? `<span style="font-size:16px;margin-right:4px;">${meta.icon}</span>` : '';

        return `<div style="flex:1;min-width:300px;border:2px solid ${borderColor};border-radius:8px;padding:14px;opacity:${opacity};transition:opacity 0.2s;${borderStyle}">
          <div class="flex gap-8 mb-12" style="align-items:center;justify-content:space-between;">
            <div class="flex gap-8" style="align-items:center;">
              ${iconHtml}<strong style="font-size:14px;">${TB.utils.esc(meta.label)}</strong>
              ${statusBadge}
              ${isHighLev ? '<span class="badge badge-red" style="font-size:9px;">RISK</span>' : ''}
            </div>
            <button class="btn btn-sm ${active ? 'btn-danger' : 'btn-success'}" data-action="toggle-group" data-group="${TB.utils.esc(g)}">
              ${active ? 'Disable' : 'Activate'}
            </button>
          </div>
          <div style="font-size:11px;font-family:monospace;color:var(--text2);line-height:1.6;">
            ${meta.stats.map(l => TB.utils.esc(l)).join('<br>')}
          </div>
        </div>`;
      }).join('') +
    '</div>';
  }

  function renderStrategyList(strategies) {
    const el = document.getElementById('strat-list');
    if (!el) return;

    if (strategies.length === 0) {
      el.innerHTML = '<div class="c-text2 text-center" style="padding:20px;">No strategies loaded</div>';
      return;
    }

    // Group strategies by group name
    const grouped = {};
    const ungrouped = [];
    for (const s of strategies) {
      if (s.group) {
        (grouped[s.group] = grouped[s.group] || []).push(s);
      } else {
        ungrouped.push(s);
      }
    }

    let html = '';

    // Render each group section
    for (const [groupName, members] of Object.entries(grouped)) {
      const meta = GROUP_META[groupName] || { label: groupName };
      const active = isGroupActive(strategies, groupName);
      const sectionOpacity = active ? '1' : '0.45';
      const statusLabel = active ? '' : ' (disabled)';

      html += `<div style="opacity:${sectionOpacity};transition:opacity 0.2s;">`;
      html += `<div style="padding:8px 12px;font-size:12px;font-weight:bold;color:var(--text2);border-bottom:2px solid var(--border);background:var(--bg2);">
        ${TB.utils.esc(meta.label)}${statusLabel}
      </div>`;
      html += members.map(renderStrategyRow).join('');
      html += '</div>';
    }

    // Ungrouped strategies
    if (ungrouped.length > 0) {
      html += ungrouped.map(renderStrategyRow).join('');
    }

    el.innerHTML = html;
  }

  function renderStrategyRow(s) {
    const roleColors = { primary: 'badge-blue', secondary: 'badge-purple', '': 'badge-gray' };
    const roleBadge = s.role ? `<span class="badge ${roleColors[s.role] || 'badge-gray'}">${TB.utils.esc(s.role.toUpperCase())}</span>` : '';
    const modeBadge = s.paper_mode ? '<span class="badge badge-yellow">PAPER</span>' : '<span class="badge badge-green">LIVE</span>';
    const statusBadge = s.status === 'ERRORED' ? '<span class="badge badge-red">ERRORED</span>' : s.status === 'DISABLED' ? '<span class="badge badge-gray">DISABLED</span>' : s.status === 'STOPPED' ? '<span class="badge badge-yellow">STOPPED</span>' : '';
    const wrWarn = s.win_rate !== null && s.win_rate < 30 ? ' c-red' : '';

    const eqBadge = s.equity_pct != null ? `<span class="badge badge-gray" style="font-size:10px;">${Math.round(s.equity_pct * 100)}% &times; ${s.leverage || '?'}x</span>` : '';

    return `<div style="padding:12px;border-bottom:1px solid var(--border);cursor:pointer;" data-action="select" data-strategy="${TB.utils.esc(s.name)}">
      <div class="flex gap-8 mb-12" style="align-items:center;flex-wrap:wrap;">
        <strong>${TB.utils.esc(s.name)}</strong>
        ${eqBadge} ${roleBadge} ${modeBadge} ${statusBadge}
      </div>
      <div style="font-size:11px;color:var(--text2);margin-bottom:6px;">Coins: ${s.coins.map(c => `<span class="badge badge-gray">${TB.utils.esc(c)}</span>`).join(' ')}</div>
      <div class="flex gap-16" style="font-size:12px;">
        <span>Trades: <strong>${s.trades}</strong></span>
        <span class="${wrWarn}">WR: <strong>${s.win_rate !== null ? s.win_rate + '%' : '\u2014'}</strong></span>
        <span class="${TB.utils.pnlClass(s.pnl)}">PnL: <strong>${TB.utils.fmtPnl(s.pnl)}</strong></span>
        ${s.position ? `<span>Pos: <span class="badge ${s.position === 'BUY' ? 'badge-green' : 'badge-red'}">${s.position}</span></span>` : ''}
      </div>
      ${s.win_rate_per_coin && Object.keys(s.win_rate_per_coin).length > 0 ?
        `<div style="font-size:10px;color:var(--text3);margin-top:4px;">WR/coin: ${Object.entries(s.win_rate_per_coin).map(([c,w]) => `${c}:${w}%`).join(' ')}</div>` : ''}
      <div class="flex gap-8 mt-8">
        <button class="btn btn-sm" data-action="toggle" data-strategy="${TB.utils.esc(s.name)}">
          ${s.status === 'DISABLED' || s.status === 'ERRORED' ? 'Enable' : 'Disable'}
        </button>
      </div>
    </div>`;
  }

  async function loadStrategies() {
    const data = await TB.api.get('/api/strategies');
    if (!data) return;

    renderGroupToggles(data);
    renderStrategyList(data);
  }

  async function selectStrategy(name) {
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
  }

  async function toggleStrategy(name) {
    const result = await TB.api.post('/api/strategies/' + encodeURIComponent(name) + '/toggle');
    if (result) {
      TB.toast.show('success', `Strategy ${name}: ${result.status}`);
      loadStrategies();
    }
  }

  async function toggleGroup(groupName) {
    const result = await TB.api.post('/api/strategies/group/' + encodeURIComponent(groupName) + '/toggle');
    if (result && !result.error) {
      const meta = GROUP_META[groupName] || { label: groupName };
      TB.toast.show('success', `${meta.label}: ${result.status}`);
      loadStrategies();
    } else if (result && result.error) {
      TB.toast.show('error', result.error);
    }
  }

  TB.pages.strategies = { mount, unmount };
})();
