# Motion pipeline rules

Use the same policy for every robot and action within a pipeline. No behavior branches based on robot names, filenames, action labels or selected frame intervals. Physical topology, axes, limits, meshes and fixed semantic/TCP frames are metadata.

Never replace the immutable original model output with a refined output. Raw priors are separate files with SHA256 references. Preserve failed results and report their status.

Only 30 Hz is currently supported by the unified runner. Do not apply the SOMA/SMPL fixed 100+100 stages to Nymeria or Gento. Do not enable new acceptance gates by default. An optional acceptance module receives final artifacts; no automatic claim of physical feasibility.

No finger-action generation. Wrist/TCP pose is the relevant endpoint. No implicit downloads, dataset uploads, cloud workers or publishing in normal runs. Keep weights and private datasets outside Git. Source math migration must be checked against provenance before changing it.

EgoAllo body reconstruction and its robot consumers use native SMPL-H only. Do not restore H-to-X conversion, body-family toggles, or fallback to converted body caches. Preserve historical outputs as evidence and regenerate native H inputs in fresh versioned workspaces.
