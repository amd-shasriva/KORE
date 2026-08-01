# Phase-0 sandbox boundary

`kore.sandbox` is the repository-side contract for candidate execution. It
provides deterministic requests, bounded transport, environment minimization,
attested verdict verification, and explicit outcome types. It does **not**
claim that an ordinary subprocess isolates hostile code.

## Modes

`trusted-subprocess` is the compatibility default. Its backend label is
`trusted-code-only`. It uses a private work directory and environment, bounded
output, wall/CPU/file/open-file rlimits, a process group, and optional cleanup
hooks. These controls reduce accidental leakage and runaway processes; code in
the subprocess still has the caller's UID, kernel access, filesystem access,
network access, and direct access to any exposed GPU device.

`external-broker` is required when either `production=true` or
`trust_level=untrusted`. The policy also requires a signed verdict and an
explicitly approved broker identity. Missing configuration, an unavailable
socket, invalid peer credentials, an unsigned/invalid verdict, or a digest
mismatch fails closed. There is no subprocess fallback.

Legacy Python driver/timing requests are non-production only. Production
requests use the declarative HSACO launch-plan model so the broker—not candidate
Python—owns module lookup, allocations, argument binding, launch order, and
device policy.

## Repository-side guarantees

- Canonical JSON and domain-separated SHA-256 task, source, policy, toolchain,
  runtime, and output digests.
- One-shot nonces with bounded replay state.
- A pluggable signed-verdict verification interface. `EphemeralLocalHMAC` is
  intentionally marked non-production and exists only for local tests.
- Length-prefixed Unix-socket frames with request, response, source, and output
  bounds.
- Linux `SO_PEERCRED` checks plus socket owner/type checks before accepting a
  broker response.
- A strict candidate environment allowlist. Ambient API keys, cloud
  credentials, proxy controls, Slurm variables, SSH variables, `LD_PRELOAD`,
  `PYTHONUSERBASE`, and inherited `PYTHONPATH` are not passed.
- Per-evaluation private `HOME`, `TMPDIR`, XDG, Triton, inductor, extension, and
  MIOpen cache paths.
- Separate candidate-error, timeout, infrastructure, policy, GPU-fault,
  GPU-quarantined, broker-unavailable, and invalid-verdict statuses.
- Validated HSACO modules, exported symbols, buffers, scalar ranges, launch
  grids, argument bindings, dependencies, scratch, and aggregate launch caps.

The client does not implement a broker server, cgroup administration, namespace
creation, seccomp/device policy, HSACO loading, GPU reset, or quarantine.

## Required production deployment

The approved external broker must run outside the training/evaluation process
and provide all host controls that this repository cannot provide:

1. Authenticate clients and bind each request to a fresh cgroup v2 subtree with
   CPU, memory, pids, I/O, and wall-time enforcement. `RLIMIT_NPROC` is not a
   replacement because it is per UID.
2. Create the required user, mount, PID, IPC, and network namespaces; install a
   deny-by-default seccomp policy; use a read-only root and bounded writable
   scratch; and run under a dedicated unprivileged identity.
3. Ignore client-supplied host paths/environment. Resolve task and HSACO content
   from broker-approved, digest-addressed stores and independently recompute all
   request digests.
4. Enforce the launch plan against device-specific allocation, dispatch,
   occupancy, scratch, stream, event, and total-work caps.
5. Capture bounded output, kill the complete cgroup on every exit path, detect
   lingering GPU holders, and perform post-run GPU health checks.
6. Quarantine a GPU after RAS/ECC, VM fault, reset, wedged queue, or failed
   cleanup. A quarantined device must not return to the scheduler until an
   operator-approved reset/health workflow succeeds.
7. Sign canonical verdict bytes with a managed production key. The client-side
   verifier must be injected through `VerdictSignatureVerifier`; a secret shared
   with the repository process is not production attestation.
8. Protect the Unix socket directory and key material, pin the expected broker
   UID/GID, rotate keys, audit request/verdict digests, and monitor broker health.

Configuration is read from `KORE_ISOLATION_MODE` and `KORE_SANDBOX_*`. External
mode requires `KORE_SANDBOX_BROKER_APPROVED=1`,
`KORE_SANDBOX_BROKER_ID`, an absolute `KORE_SANDBOX_BROKER_SOCKET`, and approved
UID/GID values. A production verifier still has to be supplied by the embedding
application; environment variables do not load signing keys.

## Shared-GPU limitation

Strong isolation on a shared, non-partitioned GPU is not available here and
cannot be manufactured by Python, process groups, containers, or cgroups. GPU
contexts share the kernel driver, firmware, memory subsystem, engines, reset
domain, and often performance state. A fault or denial-of-service workload can
affect peer contexts, and software cannot prove VRAM/confidentiality isolation
that the hardware and driver do not expose.

Production evaluation of untrusted kernels therefore requires a dedicated GPU
or a vendor-supported hardware partition with independently documented memory,
engine, fault, and reset isolation. If those capabilities are absent, the
broker must serialize onto a dedicated device and treat any GPU fault as a
whole-device quarantine event. Multi-tenant use of the same unpartitioned GPU
is explicitly outside the security guarantee.
