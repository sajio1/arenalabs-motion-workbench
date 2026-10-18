/** Exactly one owner renders static background at a time. */
export function backgroundLayers(showCloud, showMemory, hasMemory) {
  const fused = Boolean(showCloud && showMemory && hasMemory);
  return {fused, frameBackground: Boolean(showCloud && !fused)};
}
