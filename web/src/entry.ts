if ((location.pathname === '/manipulation' || location.pathname === '/localmanipulation' || location.pathname === '/assets/manipulation4d/preview.html')) {
  if (location.pathname === '/localmanipulation') history.replaceState(null,'','/manipulation');
  void import('./manipulation4d/main');
} else {
  void import('./main');
}
