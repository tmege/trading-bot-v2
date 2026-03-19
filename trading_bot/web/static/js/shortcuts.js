(function() {
  'use strict';
  window.TB = window.TB || {};

  const pages = ['dashboard', 'market', 'strategies', 'backtest', 'settings'];

  document.addEventListener('keydown', (e) => {
    const isMod = e.metaKey || e.ctrlKey;
    if (!isMod) return;

    const num = parseInt(e.key);
    if (num >= 1 && num <= 5) {
      e.preventDefault();
      location.hash = '#' + pages[num - 1];
      return;
    }

    if (e.key === 's' || e.key === 'S') {
      e.preventDefault();
      if (location.hash === '#settings' && TB._saveSettings) {
        TB._saveSettings();
      }
      return;
    }

    if (e.key === 'Escape') {
      e.preventDefault();
      const modal = document.getElementById('modal-container');
      if (modal) modal.classList.remove('visible');
    }
  });

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      const modal = document.getElementById('modal-container');
      if (modal) modal.classList.remove('visible');
    }
  });

  TB.shortcuts = { pages };
})();
