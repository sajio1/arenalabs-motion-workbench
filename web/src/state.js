export function sampleFrame(seconds, fps, frames) {
  return Math.max(0, Math.min(frames - 1, Math.floor(seconds * fps + 1e-7)));
}
export function activeRobot(defaultRobot, selected, slots) {
  return selected === null ? defaultRobot : slots[selected].robot;
}
export function visibleClips(clips, query) {
  const q = query.trim().toLowerCase();
  return clips.filter(c => [c.title,c.family,c.id,c.description,c.source_name,...motionGroup(c)].join(' ').toLowerCase().includes(q));
}
// Only a proven package fingerprint collapses uploads. Names alone are not identity.
export function robotChoices(robots) {
  const groups = new Map();
  for (const robot of robots) {
    const key = robot.report?.selection_key || robot.id;
    const old = groups.get(key);
    if (!old || (robot.conversion_ready && !old.conversion_ready)) groups.set(key,robot);
  }
  return [...groups.values()];
}
export function motionGroup(clip) {
  if (clip.metadata?.['Activity category']) return [clip.metadata['Activity category'],clip.metadata['Activity subclass'] || 'Session excerpts'];
  const raw = String(clip.metadata?.Task || clip.family || '').replace(/^ego_\d+_/, '');
  const label = raw.replace(/_/g,' ').replace(/\b\w/g,c=>c.toUpperCase());
  if (clip.page === 'locomotion') {
    const text = (clip.source_name+' '+raw).toLowerCase();
    for (const [pattern,family,subclass] of [[/dance/,'Dance','Dance sequences'],[/jump/,'Locomotion','Jumping'],[/walk/,'Locomotion','Walking'],[/run|jog/,'Locomotion','Running & jogging'],[/crawl/,'Locomotion','Crawling'],[/crouch|squat|sit|stand/,'Posture & transitions','Posture changes'],[/stretch|lunge|exercise/,'Exercise','Mobility & exercise'],[/idle/,'Stationary','Idle & standing']]) if(pattern.test(text))return [family,subclass];
    return ['Other movements',raw==='Cached excerpts'?'Reference excerpts':label];
  }
  if (/wash|clean|wipe|sweep|declutter|dry_hands/.test(raw)) return ['Cleaning & housekeeping',label];
  if (/food|bread|egg|fruit|sandwich|chopstick|kitchen|pour|scoop|ice/.test(raw)) return ['Cooking & food preparation',label];
  if (/fold|cloth|shoelace|knit|braid|zip|rubberband/.test(raw)) return ['Clothing & textiles',label];
  if (/lego|jenga|domino|puzzle|game|cards|dice|chess|reversi|mancala|connect_four|rubiks|toy|coin|ball/.test(raw)) return ['Games & play',label];
  if (/assemble|screw|fixture|structures|tiles|furniture/.test(raw)) return ['Assembly & tools',label];
  if (/write|paper|paint|color|keyboard|stamp|staple|piano/.test(raw)) return ['Desk work & crafts',label];
  if (/hair|makeup|tooth|glasses/.test(raw)) return ['Personal care',label];
  if (/charge|airpods|device|plug|usb|batteries|remote/.test(raw)) return ['Devices & electronics',label];
  if (/pick|place|stack|insert|remove|open|close|lid|wrap|stock|lock|measure/.test(raw)) return ['Object handling & storage',label];
  return ['Other activities',/Cached excerpts|Greenwich|^source$/.test(raw)?'Reference excerpts':label];
}
export function safeText(value) {
  return String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
