# Model files, Hugging Face and deployment access

## Do users need our weights?

| Usage | Who needs the weights? | Does the end user need SSH? |
|---|---|---|
| Use our hosted website | The cloud worker/service owner | No |
| Call our hosted HTTP API | The cloud worker/service owner | No; use the application's workspace/session |
| Reproduce locally or deploy on another server | That installation's administrator | Only their own server's administrative access |
| Contribute code | Maintainers deploy reviewed commits | Contributors do not need production SSH |

Publishing the GitHub repository does not require publishing the model weights.
If the goal is independently reproducible local/self-hosted inference, users
need a documented authorized way to obtain every required model. Hugging Face
is a suitable distribution channel for first-party weights; it is not required
for hosted users and it is not a way to bypass upstream model terms.

## Model inventory for the current workbench

| Component | Needed files / identity | Distribution approach |
|---|---|---|
| AlphaMotion / Greenwich Prime | `config.json`, `model.safetensors`, compatible tokenizer/semantic model dependencies and code version | Publish the first-party checkpoint if the owner has distribution rights, or offer gated/private access |
| EgoAllo | The upstream denoiser release used by the adapters (`egoallo_checkpoint_april13/checkpoints_3000000`) | Link to the actual upstream release and record a hash/revision; do not assume a code license alone describes every external asset |
| Extended SMPL-H | 16-beta neutral model loaded as `model.npz` | Obtain through the authorized upstream download/license process |
| SMPL-X | Neutral model loaded as `SMPLX_NEUTRAL.npz`, supporting the configured shape dimensions | Obtain through the authorized upstream download/license process |
| DA3 | `depth-anything/DA3NESTED-GIANT-LARGE-1.1` | Download from the official upstream model with its terms, pin revision |
| Qwen vision-language model | `Qwen/Qwen2.5-VL-7B-Instruct` and its processor/tokenizer | Official upstream model; pin revision and dependency versions |
| SAM 3.1 | Authorized multiplex checkpoint (`sam3.1_multiplex.pt`) and matching SAM source | Each installation needs authorized access; link to upstream instructions |
| Robot and hand assets | MJCF/URDF, meshes, limits, joint roles, reference frames, Wuji mapping | Track the asset owner's terms and mechanical metadata separately |

The current repository excludes weights/private data. It also does not include
a fully pinned installer for all upstream model environments. The AlphaMotion
runtime sets offline loading after configuration; dependencies must be obtained
before a normal inference job. A missing weight must fail clearly, not trigger
an unannounced download in the request path.

## Recommended first-party Hugging Face release

Release only the model you own or have explicit permission to distribute. A
minimal versioned repository should contain:

```text
README.md                 model card
LICENSE                   chosen weight license, when authorized
config.json               exact runtime architecture/config
model.safetensors         exact compatible checkpoint
checksums.sha256           file integrity
runtime-compatibility.md   tested code commit, semantic dependencies, versions
```

The model card should state the purpose, required input geometry and six-point
visibility, single-frame inference, supported units/rates, training/evaluation
scope, known failure cases and intended use. Describe raw motion versus refined
motion clearly. Do not imply all robot embodiments or physical tasks have been
validated. Record the code commit and immutable Hub revision used for a release.

Hugging Face supports public, private and gated model repositories. Gating can
require individual requests and either automatic or manual approval. Select
that release policy deliberately. The workbench currently loads local files;
publishing a model does not automatically add a downloader to this code.

Example commands for an **administrator to adapt**, not commands run by this
repository during inference:

```bash
# Requires an authenticated account and an authorized, chosen target repository.
# The staging directory must contain only the intended first-party release.
hf upload YOUR_ORG/YOUR_MODEL ./first-party-release --repo-type model

# Consumers download an explicitly pinned revision during installation.
hf download YOUR_ORG/YOUR_MODEL --revision RELEASE_COMMIT   --local-dir /path/to/private/weights/prime
```

Point the runtime JSON's `weights` field at that local directory. Never upload
a whole cloud working directory: it can contain other licensed checkpoints,
private recordings, generated episodes, caches or credentials. The current
request asks whether a release is needed; no model files have been uploaded to
Hugging Face as part of this audit.

## Third-party body-model terms

The official SMPL-X page describes a personal, non-transferable research license,
restricts third-party availability/distribution, and provides a separate route
for commercial licensing. Do not bundle SMPL-X or derived model templates into
a first-party weight repository just because they are on the same machine.
SMPL-H/MANO and other upstream assets require their own applicable terms.

A public GitHub repository is not itself a license grant. This repository has
not selected a project-wide first-party code or model license. Its individual
third-party notices remain in force. Choose explicit first-party licenses and
resolve upstream permissions before advertising an unrestricted reusable or
commercial model bundle.

## Cloud access model

A website user submits a recording over HTTPS, polls a job and downloads their
result. Models stay on the worker. A developer submits code through GitHub; a
trusted maintainer or CI deployment account installs it. SSH is an administrator
capability, not an inference API or a necessary recording-upload mechanism.

Do not give users the owner's private SSH key, a shared cloud shell, or broad
write access to worker model/config directories. For a production multi-user
service, add account authentication, per-owner job authorization, quotas,
rate/size/concurrency limits, storage retention and operator monitoring. The
current browser workspace cookie provides basic isolation, not these full
service controls. A GitHub push does not currently trigger an automatic cloud
deployment workflow.

## Primary references

- [Hugging Face gated models](https://huggingface.co/docs/hub/models-gated)
- [Hugging Face model cards](https://huggingface.co/docs/hub/model-cards)
- [Hugging Face upload guide](https://huggingface.co/docs/huggingface_hub/guides/upload)
- [Official SMPL-X model license](https://smpl-x.is.tue.mpg.de/modellicense.html)
- [Official EgoAllo repository](https://github.com/brentyi/egoallo)
