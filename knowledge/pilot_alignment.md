# pilot alignment

## what the pilot is

The LTRC Infrastructure Upgrade Pilot is a 6-month institutional programme to build shared GPU infrastructure for the lab. Phase A (months 1–3) covers hardware setup, Kubernetes, shared storage (NAS/PFS), a Docker registry, a shared HuggingFace model cache, onboarding documentation, and logging and telemetry. Phase B (months 4–6) covers rollout to users, usage tracking, and iterative improvements based on real use. The pilot explicitly funds NAS, Kubernetes, and a Docker registry as Phase A deliverables — these are not long-term or post-serving items.


## where stage0 fits

| stage0 deliverable | pilot dependency it satisfies |
|---|---|
| Verified GPU env on Turing | Phase A: standardized environment baseline |
| Docker image with pinned deps | Phase A: Docker registry input; burst-mode artefact |
| Benchmark numbers (tokens/sec, GPU util, TTFT) | Phase A: telemetry baseline; Phase B: usage tracking |
| vLLM endpoint with Prometheus metrics | Phase A: API access to inference; telemetry |
| InfiniBand investigation notes | Phase A: network configuration input |


## what is a parallel track (not my task)

- NAS purchase and setup (Phase A prerequisite, currently blocked on hardware — track status with supervisor)
- Kubernetes cluster setup (Phase A, run by broader team — your job is K8s-compatible Docker images)
- Docker registry setup (Phase A — your job is to produce images that go into it)
- Onboarding documentation (Phase B — your env scripts and README become inputs)


## k8s-compatibility checklist

Use this when writing any Dockerfile or serving script:

- [ ] No hardcoded absolute paths — all paths via environment variables
- [ ] `HF_HOME`, `PIP_CACHE_DIR`, `CONDA_PKGS_DIRS` set via `ENV` in Dockerfile
- [ ] vLLM (or any serving process) exposes a `/health` or `/metrics` endpoint
- [ ] Prometheus metrics endpoint wired and documented
- [ ] Container runs as non-root user
- [ ] No secrets or credentials baked into the image


## phase timeline

| Phase | Months | What happens | Your involvement |
|---|---|---|---|
| Phase A | 1–3 | Hardware setup, NAS/PFS, Kubernetes, Docker registry, shared HF cache, onboarding, logging and telemetry | Stage0 artefacts (env, Docker image, benchmarks, vLLM endpoint) are direct inputs; IB investigation feeds network config |
| Phase B | 4–6 | Rollout to lab users, usage tracking, iterative improvements | Env scripts and README feed onboarding docs; benchmark baseline feeds usage tracking and optimization work |
