// Shared by the workspace build and the standalone deployment patch.
if (!document.documentElement.hasAttribute('data-generation-dismiss-installed')) {
  document.documentElement.setAttribute('data-generation-dismiss-installed', '');
  let timer, trackedPanel, trackedState;
  const terminal = panel => panel && ['completed', 'failed', 'interrupted', 'cancelled'].includes(panel.dataset.state);
  const clear = () => { clearTimeout(timer); timer = undefined; };
  const dismiss = () => {
    const panel = document.getElementById('generation-feedback');
    if (terminal(panel)) panel.hidden = true;
    clear();
  };
  const sync = () => {
    const panel = document.getElementById('generation-feedback');
    const state = panel && !panel.hidden ? panel.dataset.state : undefined;
    if (panel === trackedPanel && state === trackedState) return;
    clear(); trackedPanel = panel; trackedState = state;
    if (terminal(panel) && !panel.hidden) timer = setTimeout(dismiss, state === 'completed' ? 3000 : 8000);
  };
  new MutationObserver(sync).observe(document.documentElement, {
    subtree: true, childList: true, attributes: true, attributeFilter: ['hidden', 'data-state'],
  });
  document.addEventListener('pointerdown', event => {
    const panel = document.getElementById('generation-feedback');
    if (terminal(panel) && !panel.contains(event.target)) dismiss();
  });
  document.addEventListener('keydown', event => { if (event.key === 'Escape') dismiss(); });
  sync();
}
