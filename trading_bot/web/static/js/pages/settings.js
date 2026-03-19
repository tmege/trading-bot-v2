(function() {
  'use strict';
  window.TB = window.TB || {};
  TB.pages = TB.pages || {};

  let settingsData = null;

  function mount(container) {
    container.innerHTML = `
      <div class="section-header">
        <h2>Settings</h2>
        <button class="btn btn-primary" id="btn-save-settings">Save (Cmd+S)</button>
      </div>
      <div id="settings-loading" class="c-text2">Loading settings...</div>
      <div id="settings-content" style="display:none;"></div>
    `;

    document.getElementById('btn-save-settings').onclick = saveSettings;
    TB._saveSettings = saveSettings;

    loadSettings();
  }

  function unmount() {
    TB._saveSettings = null;
    settingsData = null;
  }

  async function loadSettings() {
    settingsData = await TB.api.get('/api/settings');
    if (!settingsData || settingsData.error) {
      document.getElementById('settings-loading').textContent = 'Failed to load settings';
      return;
    }

    document.getElementById('settings-loading').style.display = 'none';
    const el = document.getElementById('settings-content');
    el.style.display = '';

    renderSettings(el);
  }

  function renderSettings(el) {
    const s = settingsData;

    el.innerHTML = `
      <div class="card mb-16">
        <div class="card-header">Risk Parameters</div>
        <div class="card-body">
          <div class="form-row">
            <div class="form-group" style="flex:1;">
              <label>Daily Loss % (1-50)</label>
              <input id="risk-daily-loss" type="number" class="input w-full" min="1" max="50" step="0.5" value="${s.risk.daily_loss_pct}">
            </div>
            <div class="form-group" style="flex:1;">
              <label>Emergency Close % (1-50)</label>
              <input id="risk-emergency" type="number" class="input w-full" min="1" max="50" step="0.5" value="${s.risk.emergency_close_pct}">
            </div>
            <div class="form-group" style="flex:1;">
              <label>Max Position % (10-10000)</label>
              <input id="risk-max-pos" type="number" class="input w-full" min="10" max="10000" step="10" value="${s.risk.max_position_pct}">
            </div>
            <div class="form-group" style="flex:1;">
              <label>Max Leverage (1-50)</label>
              <input id="risk-max-lev" type="number" class="input w-full" min="1" max="50" step="1" value="${s.risk.max_leverage}">
            </div>
          </div>
        </div>
      </div>

      <div class="card mb-16">
        <div class="card-header">Strategies</div>
        <div class="card-body" id="settings-strategies"></div>
      </div>
    `;

    renderStrategies();
  }

  function renderStrategies() {
    const el = document.getElementById('settings-strategies');
    if (!el || !settingsData) return;

    el.innerHTML = settingsData.strategies.map((strat, idx) => `
      <div class="mb-20" style="padding-bottom:16px;border-bottom:1px solid var(--border);">
        <div class="flex gap-8 mb-12" style="align-items:center;">
          <strong>${TB.utils.esc(strat.file)}</strong>
          ${strat.role ? `<span class="badge badge-blue">${TB.utils.esc(strat.role.toUpperCase())}</span>` : ''}
        </div>
        <div class="form-group mb-12">
          <label>Coins</label>
          <div class="pills" id="strat-coins-${idx}">
            ${strat.coins.map(c => `<span class="pill active" data-coin="${TB.utils.esc(c)}" data-idx="${idx}" style="cursor:default;opacity:0.8;">${TB.utils.esc(c)}</span>`).join('')}
          </div>
        </div>
        <div class="form-row">
          <div class="form-group">
            <label>Paper Mode</label>
            <label class="toggle">
              <input type="checkbox" id="strat-paper-${idx}" ${strat.paper_mode ? 'checked' : ''}>
              <span class="toggle-slider"></span>
            </label>
          </div>
          <div class="form-group">
            <label>Paper Balance</label>
            <input id="strat-balance-${idx}" type="number" class="input" style="width:100px;" value="${strat.paper_balance || 500}">
          </div>
        </div>
      </div>
    `).join('');
  }

  async function saveSettings() {
    if (!settingsData) return;

    const risk = {
      daily_loss_pct: parseFloat(document.getElementById('risk-daily-loss').value),
      emergency_close_pct: parseFloat(document.getElementById('risk-emergency').value),
      max_position_pct: parseFloat(document.getElementById('risk-max-pos').value),
      max_leverage: parseInt(document.getElementById('risk-max-lev').value),
    };

    const strategies = settingsData.strategies.map((strat, idx) => {
      const paperMode = document.getElementById('strat-paper-' + idx).checked;
      const paperBalance = parseFloat(document.getElementById('strat-balance-' + idx).value) || 500;

      return {
        file: strat.file,
        role: strat.role,
        coins: strat.coins,
        paper_mode: paperMode,
        paper_balance: paperBalance,
      };
    });

    const result = await TB.api.put('/api/settings', { risk, strategies });
    if (result && result.status === 'ok') {
      TB.toast.show('success', 'Configuration saved');
      if (result.restart_required) {
        TB.toast.show('warning', 'Restart bot to apply changes', true);
      }
    } else {
      TB.toast.show('error', result ? (result.error || 'Save failed') : 'Save failed');
    }
  }

  TB.pages.settings = { mount, unmount };
})();
