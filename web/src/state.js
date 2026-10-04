export function sampleFrame(seconds, fps, frames) {
  return Math.max(0, Math.min(frames - 1, Math.floor(seconds * fps + 1e-7)));
}
export function activeRobot(defaultRobot, selected, slots) {
  return selected === null ? defaultRobot : slots[selected].robot;
}
export function visibleClips(clips, query) {
  const q = query.trim().toLowerCase();
  return clips.filter(c => [c.title,c.family,c.id,c.description,c.source_name].join(' ').toLowerCase().includes(q));
}
export function safeText(value) {
  return String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
