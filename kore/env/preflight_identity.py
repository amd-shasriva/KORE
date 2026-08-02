"""Produce the preflight runtime identity that authorizes replay caching.

``kore.env.evaluation_contract`` refuses to mark an evaluation cacheable unless
it is handed a validated *preflight runtime identity*. Nothing produced one, so
``cacheable_context`` was permanently False, ``replay_hits`` permanently 0, and
every duplicate candidate in a GRPO group paid a full ~8.7 s GPU evaluation.
This module is the missing producer.

Why this cannot be done inside ``KoreEnv``
------------------------------------------
The identity is the *evidence* that authorizes replaying a measurement. If
``KoreEnv`` minted it at evaluation time, the evidence would be manufactured by
the very process, at the very instant, and from the very hardware state that it
then authorizes. Whatever the machine happened to look like would be recorded as
"validated and stable" by construction, the check could never fail, and
``cacheable_context`` would degenerate into an expensive way of writing ``True``.
The guarantee "an observation may only be replayed against proof the hardware
and runtime were validated" would mean nothing.

So the producer lives here, and it is a preflight in the literal sense:

* **Earlier.** It runs in a distinct startup phase - an operator command, or one
  call at the top of :func:`kore.policy.grpo.train_grpo` - strictly before any
  candidate is evaluated. ``KoreEnv`` only ever *consumes*; it never mints.
* **Independent.** It knows nothing about candidates, tasks, or observations.
  Its inputs are the machine and the toolchain.
* **Falsifiable.** It does real work that really fails: it enumerates HIP out of
  process, joins those devices to DRM/sysfs by PCI BDF, requires the UUIDs from
  those two independent sources to agree, runs an exactly-checkable computation
  on each GPU, and re-probes to confirm the answer is reproducible. A GPU that
  fails any of it simply gets no identity, and evaluations on it stay uncached.
* **Evidence, not assertion.** The result is a content-addressed artifact that
  can be written to disk, audited, and handed to workers that did not produce it.

Re-checking that evidence at evaluation time is not circular. Generating a
certificate for yourself is; confirming that the world still matches a
certificate someone else established earlier is exactly what makes it evidence.

What the identity is bound to, and why
--------------------------------------
An identity must invalidate itself when the thing it attests to changes, so each
identity declares the live facts it was validated against. The consumer
re-verifies every declared binding on every evaluation and fails closed on drift:

``boot_id``
    HIP's device enumeration order is fixed for the life of a boot but may be
    renumbered across boots - this box already enumerates ordinal 0 as PCI bus
    0x78 and ordinal 1 as bus 0x08, so ordinal order is not BDF order. Binding to
    the boot means "HIP ordinal 5 is this physical card" stays true for exactly
    as long as it is checkable without re-initializing HIP in the consumer.
``toolchain_sha256``
    Compilers, ROCm, torch/triton/aiter. A measurement made under one toolchain
    is not evidence about another, and the preflight only validated the one that
    was live when it ran.
``core_code_sha256``
    The evaluator, reward, roofline and parser sources that decide a verdict.

Hardware identity, the GPU target, and the selected GPU are mandatory and are
checked for every identity regardless of what it declares.

How it reaches the rollout workers
----------------------------------
``gpu_target`` and ``selected_gpu`` are per-evaluation values computed inside
``KoreEnv``, and under distributed GRPO each rank benches on its own physical
GPU, so one flat identity exported for a whole multi-GPU run would be wrong for
every rank but one. The artifact is therefore a *bundle* holding one identity per
validated GPU, and the consumer selects its own entry by the (selected GPU, GPU
target) pair that only it knows. One value, exported once, is correct for every
rank - and a rank whose GPU is not in the bundle gets no identity rather than
somebody else's.

Two delivery paths, both supported:

* out of process - ``python -m kore.env.preflight_identity`` writes the artifact
  and prints a shell ``export`` line for ``KORE_PREFLIGHT_RUNTIME_IDENTITY``,
  which every worker inherits. This is the strongest form: the evidence is
  established by a different process than the one it authorizes.
* in process - :func:`establish_preflight_identity` at the top of training, then
  :func:`preflight_identity_for` when each rollout env is constructed.

Identities are content-addressed: they contain no timestamps, PIDs or run ids,
only validated facts. Two preflights of an unchanged machine mint byte-identical
identities, so a persistent replay cache survives a restart; anything that
actually changes - a reboot, a toolchain bump, an edited evaluator, a different
card behind the ordinal - mints a different identity and strands the old entries.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from kore.env.evaluation_contract import (
    PREFLIGHT_IDENTITY_BUNDLE_KIND,
    PREFLIGHT_IDENTITY_ENV,
    boot_identity,
    core_code_digest,
    toolchain_digest,
)
from kore.policy.resources import (
    HIPProbeError,
    ResourcePreflightError,
    atomic_write_json,
    collect_amd_gpu_devices,
    probe_hip_inventory,
)


IDENTITY_VERSION = 1
PRODUCER = "kore.env.preflight_identity"
PRODUCER_VERSION = 1
IDENTITY_ARTIFACT_ENV = "KORE_PREFLIGHT_IDENTITY_FILE"
IDENTITY_GPUS_ENV = "KORE_PREFLIGHT_IDENTITY_GPUS"

# Every binding here must be re-verifiable by the contract from live state
# without initializing a GPU. A binding the consumer cannot check is refused
# rather than trusted, so this list is deliberately short.
DECLARED_BINDINGS = ("boot_id", "core_code_sha256", "toolchain_sha256")

_LOCK = threading.RLock()
_PROCESS_BUNDLE: Optional[dict[str, Any]] = None


class PreflightIdentityError(RuntimeError):
    """Raised when a preflight was requested but could not be established."""


def _base_gpu_target(gcn_arch_name: str) -> str:
    """``gfx950:sramecc+:xnack-`` -> ``gfx950``, matching Task.gpu_target."""
    return str(gcn_arch_name or "").split(":", 1)[0].strip()


def _requested_gpus(
    gpus: Optional[Sequence[int | str]], environ: Mapping[str, str]
) -> Optional[tuple[int, ...]]:
    raw: Any = gpus
    if raw is None:
        raw = environ.get(IDENTITY_GPUS_ENV) or None
    if raw is None:
        # None means "inherit this process's own visible-device mask", so the
        # preflight can never certify a GPU the caller was not already given.
        return None
    if isinstance(raw, str):
        raw = [piece.strip() for piece in raw.split(",") if piece.strip()]
    try:
        ordinals = tuple(int(value) for value in raw)
    except (TypeError, ValueError) as exc:
        raise PreflightIdentityError(f"invalid GPU list {gpus!r}") from exc
    if not ordinals or len(set(ordinals)) != len(ordinals):
        raise PreflightIdentityError(f"invalid GPU list {gpus!r}")
    return ordinals


def _drm_join(
    inventory: tuple[Mapping[str, Any], ...], environ: Mapping[str, str]
) -> dict[int, Any]:
    """Join HIP devices to DRM/sysfs cards by PCI BDF.

    Delegated to :mod:`kore.policy.resources` precisely because it pairs the two
    sources by BDF instead of by discovery order, and keeps the UUID each source
    reported separately so a mismatch is visible rather than silently averaged.
    """
    try:
        devices = collect_amd_gpu_devices(
            hip_inventory=tuple(dict(entry) for entry in inventory),
            environ=dict(environ),
        )
    except ResourcePreflightError as exc:
        raise PreflightIdentityError(f"DRM/HIP inventory join failed: {exc}") from exc
    return {
        int(device.hip_ordinal): device
        for device in devices
        if isinstance(device.hip_ordinal, int)
    }


def _probe_twice(
    ordinals: Optional[tuple[int, ...]], environ: Mapping[str, str]
) -> tuple[tuple[dict[str, Any], ...], dict[int, str]]:
    """Probe HIP twice and report which ordinals were reproducible.

    A single probe cannot tell a stable machine from one that is renumbering
    devices or intermittently failing to bring a card up, and "stable" is a
    claim this identity actually makes.
    """
    first = probe_hip_inventory(
        ordinals=ordinals, verify_compute=True, environ=dict(environ)
    )
    second = probe_hip_inventory(
        ordinals=ordinals, verify_compute=True, environ=dict(environ)
    )

    def _identity_fields(entry: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            entry.get("pci_bdf"),
            entry.get("uuid"),
            entry.get("gcn_arch_name"),
            entry.get("total_hbm_bytes"),
            entry.get("compute_check"),
        )

    repeat = {
        int(entry["hip_ordinal"]): _identity_fields(entry) for entry in second
    }
    unstable: dict[int, str] = {}
    for entry in first:
        ordinal = int(entry["hip_ordinal"])
        if ordinal not in repeat:
            unstable[ordinal] = "disappeared between repeated probes"
        elif repeat[ordinal] != _identity_fields(entry):
            unstable[ordinal] = "reported different identity between probes"
    return first, unstable


def _mint_identity(
    entry: Mapping[str, Any],
    device: Any,
    *,
    boot_id: str,
    toolchain_sha256: str,
    core_code_sha256: str,
) -> dict[str, Any]:
    """Build one GPU's identity from facts that were actually validated."""
    return {
        "identity_version": IDENTITY_VERSION,
        "validated": True,
        "stable": True,
        "hardware": {
            # The physical card, cross-confirmed by DRM/sysfs and HIP. Pointing
            # the same ordinal at different silicon changes this, hence the key.
            "id": str(entry["uuid"]),
            "gpu_target": _base_gpu_target(str(entry.get("gcn_arch_name", ""))),
            # The absolute HIP ordinal string KoreEnv writes into the evaluator
            # subprocess's HIP_VISIBLE_DEVICES.
            "selected_gpu": str(entry["hip_ordinal"]),
            "pci_bdf": str(entry["pci_bdf"]),
            "drm_card": str(device.drm_card),
            "render_node": str(device.render_node),
            "gcn_arch_name": str(entry.get("gcn_arch_name", "")),
            "product_name": str(entry.get("name", "")),
            "total_hbm_bytes": int(entry.get("total_hbm_bytes", 0)),
            "multi_processor_count": int(entry.get("multi_processor_count", 0)),
        },
        "runtime": {
            "producer": PRODUCER,
            "producer_version": PRODUCER_VERSION,
            "boot_id": boot_id,
            "toolchain_sha256": toolchain_sha256,
            "core_code_sha256": core_code_sha256,
            "torch_version": str(entry.get("torch_version", "")),
            "hip_version": str(entry.get("hip_version", "")),
            "checks": {
                "hip_enumerated": True,
                "compute_verified": True,
                "repeat_probe_stable": True,
                "drm_bdf_join": True,
                "drm_hip_uuid_agree": True,
            },
        },
        "bindings": list(DECLARED_BINDINGS),
    }


def build_preflight_identity_bundle(
    *,
    config: Any = None,
    gpus: Optional[Sequence[int | str]] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> dict[str, Any]:
    """Run the preflight and mint one identity per GPU that passed it.

    Raises :class:`PreflightIdentityError` when the *host* could not be
    validated (no readable boot id, an unstable toolchain or core-code
    fingerprint, an unusable HIP probe). Individual GPUs that fail are recorded
    in ``rejected`` and simply get no identity; evaluations on them stay
    uncached, which is the correct fail-closed outcome rather than an error.
    """
    env = dict(os.environ if environ is None else environ)
    if config is None:
        from kore.config import CONFIG

        config = CONFIG

    boot_id = boot_identity()
    if not boot_id:
        raise PreflightIdentityError(
            "boot identity is unreadable, so no identity could be bound to this "
            "boot; replay must stay disabled"
        )
    toolchain = toolchain_digest(config)
    if toolchain.get("state") != "stable":
        raise PreflightIdentityError(
            "toolchain fingerprint is unstable, so the runtime this preflight "
            "would attest to is not even self-consistent"
        )
    core_code = core_code_digest()
    if core_code.get("state") != "stable":
        raise PreflightIdentityError(
            "core evaluator code fingerprint is unstable; refusing to attest to "
            "evaluator semantics that are changing underneath the preflight"
        )

    ordinals = _requested_gpus(gpus, env)
    started = time.time()
    inventory, unstable = _probe_twice(ordinals, env)
    joined = _drm_join(inventory, env)

    identities: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for entry in sorted(inventory, key=lambda item: int(item["hip_ordinal"])):
        ordinal = int(entry["hip_ordinal"])
        reason = unstable.get(ordinal)
        device = joined.get(ordinal)
        if reason is None and device is None:
            reason = "no DRM/sysfs card joined to this HIP device by PCI BDF"
        elif reason is None and str(device.uuid) != str(entry.get("uuid", "")):
            reason = (
                f"DRM UUID {device.uuid!r} and HIP UUID {entry.get('uuid')!r} "
                "disagree for the same PCI BDF"
            )
        elif reason is None and entry.get("compute_check") != "pass":
            reason = f"compute check did not pass: {entry.get('compute_check')}"
        elif reason is None and not _base_gpu_target(
            str(entry.get("gcn_arch_name", ""))
        ):
            reason = "HIP reported no GPU architecture"
        if reason is not None:
            rejected.append({"selected_gpu": str(ordinal), "reason": reason})
            continue
        identities.append(
            _mint_identity(
                entry,
                device,
                boot_id=boot_id,
                toolchain_sha256=str(toolchain["sha256"]),
                core_code_sha256=str(core_code["sha256"]),
            )
        )

    return {
        "identity_version": IDENTITY_VERSION,
        "kind": PREFLIGHT_IDENTITY_BUNDLE_KIND,
        "producer": PRODUCER,
        "producer_version": PRODUCER_VERSION,
        "identities": identities,
        "rejected": rejected,
        # Bundle-level only. The consumer selects a single identity out of the
        # bundle, so nothing here can reach the replay key - which is why
        # provenance may carry wall-clock time while an identity may not.
        "provenance": {
            "hostname": socket.gethostname(),
            "established_at_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)
            ),
            "probe_seconds": round(time.time() - started, 3),
            "python": platform.python_version(),
            "requested_gpus": list(ordinals) if ordinals is not None else None,
        },
    }


def establish_preflight_identity(
    *,
    config: Any = None,
    gpus: Optional[Sequence[int | str]] = None,
    environ: Optional[Mapping[str, str]] = None,
    artifact_path: Optional[str | Path] = None,
    export_environment: bool = False,
    required: bool = False,
) -> Optional[dict[str, Any]]:
    """Establish this process's preflight identity bundle, once.

    Adopts an already-exported bundle when a launcher established one out of
    process, which is the stronger arrangement and must not be overridden by a
    second, in-process opinion. Returns ``None`` instead of raising unless
    ``required``: a host where the preflight cannot run is a host that gets no
    replay caching, not a host that cannot train.
    """
    global _PROCESS_BUNDLE

    env = os.environ if environ is None else environ
    with _LOCK:
        if _PROCESS_BUNDLE is not None:
            return _PROCESS_BUNDLE
        adopted = _bundle_from_environment(env)
        if adopted is not None:
            _PROCESS_BUNDLE = adopted
            return adopted

    try:
        bundle = build_preflight_identity_bundle(
            config=config, gpus=gpus, environ=env
        )
    except (PreflightIdentityError, HIPProbeError, ResourcePreflightError, OSError) as exc:
        if required:
            raise PreflightIdentityError(str(exc)) from exc
        return None

    destination = artifact_path or env.get(IDENTITY_ARTIFACT_ENV)
    if destination:
        try:
            atomic_write_json(destination, bundle)
        except OSError as exc:
            if required:
                raise PreflightIdentityError(
                    f"preflight identity artifact could not be written: {exc}"
                ) from exc
    if export_environment:
        os.environ[PREFLIGHT_IDENTITY_ENV] = json.dumps(
            bundle, sort_keys=True, separators=(",", ":")
        )
    with _LOCK:
        if _PROCESS_BUNDLE is None:
            _PROCESS_BUNDLE = bundle
        return _PROCESS_BUNDLE


def _bundle_from_environment(
    environ: Mapping[str, str],
) -> Optional[dict[str, Any]]:
    for name, loader in (
        (PREFLIGHT_IDENTITY_ENV, json.loads),
        (IDENTITY_ARTIFACT_ENV, lambda path: json.loads(Path(path).read_text())),
    ):
        raw = environ.get(name)
        if not raw:
            continue
        try:
            payload = loader(raw)
        except (OSError, ValueError):
            continue
        if isinstance(payload, dict) and isinstance(payload.get("identities"), list):
            return payload
    return None


def resolve_selected_gpu(
    gpu: Any = None, *, environ: Optional[Mapping[str, str]] = None
) -> str:
    """The absolute GPU id string ``KoreEnv`` will select for this rollout.

    Mirrors ``KoreEnv._gpu_selection``: an explicit ``gpu`` is used as given,
    otherwise the first entry of the inherited HIP mask, otherwise ``"0"``. It
    exists so callers can look an identity up *before* constructing the env.
    Being wrong here can only cost a cache hit, never cause a wrong one: the
    contract compares the identity against the selection KoreEnv actually
    computed, so a disagreement fails closed.
    """
    if gpu is not None:
        return str(gpu)
    env = os.environ if environ is None else environ
    inherited = env.get("HIP_VISIBLE_DEVICES")
    if inherited is None:
        return "0"
    return str(inherited).split(",")[0].strip()


def preflight_identity_for(
    selected_gpu: Any,
    gpu_target: Optional[str],
    *,
    bundle: Optional[Mapping[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    """Look up this GPU's identity. A pure lookup - it never probes anything.

    Returns ``None`` when no preflight was established or when this exact
    (GPU, architecture) pair was not validated, so an unprefighted run behaves
    exactly as before: correct, uncached, and slow.
    """
    if selected_gpu is None or not gpu_target:
        return None
    with _LOCK:
        source = bundle if bundle is not None else _PROCESS_BUNDLE
    if source is None:
        return None
    matches = [
        identity
        for identity in source.get("identities", [])
        if isinstance(identity, Mapping)
        and str(identity.get("hardware", {}).get("selected_gpu")) == str(selected_gpu)
        and identity.get("hardware", {}).get("gpu_target") == str(gpu_target)
    ]
    # Exactly one, or nothing: an ambiguous bundle is not evidence about which
    # of two conflicting entries describes this GPU.
    return dict(matches[0]) if len(matches) == 1 else None


def reset_preflight_identity() -> None:
    """Test/support hook: forget the process-established bundle."""
    global _PROCESS_BUNDLE
    with _LOCK:
        _PROCESS_BUNDLE = None


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m kore.env.preflight_identity",
        description=(
            "Validate this host's GPUs and runtime and emit the preflight "
            "identity bundle that authorizes KORE replay caching."
        ),
    )
    parser.add_argument(
        "--gpus",
        default=None,
        help="comma-separated absolute HIP ordinals (default: this process's "
             "visible devices)",
    )
    parser.add_argument(
        "--out", default=None, help="write the identity bundle artifact here"
    )
    parser.add_argument(
        "--export", action="store_true",
        help=f"print a shell export line for {PREFLIGHT_IDENTITY_ENV}",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        bundle = build_preflight_identity_bundle(gpus=args.gpus)
    except (PreflightIdentityError, HIPProbeError, ResourcePreflightError) as exc:
        print(f"preflight identity could not be established: {exc}", file=sys.stderr)
        return 1

    if args.out:
        atomic_write_json(args.out, bundle)
    if args.export:
        payload = json.dumps(bundle, sort_keys=True, separators=(",", ":"))
        print(f"export {PREFLIGHT_IDENTITY_ENV}={payload!r}")
    else:
        print(json.dumps(bundle, indent=2, sort_keys=True))
    for entry in bundle["rejected"]:
        print(
            f"GPU {entry['selected_gpu']} was NOT validated: {entry['reason']}",
            file=sys.stderr,
        )
    return 0 if bundle["identities"] else 1


__all__ = [
    "DECLARED_BINDINGS",
    "IDENTITY_ARTIFACT_ENV",
    "IDENTITY_GPUS_ENV",
    "IDENTITY_VERSION",
    "PRODUCER",
    "PRODUCER_VERSION",
    "PreflightIdentityError",
    "build_preflight_identity_bundle",
    "establish_preflight_identity",
    "preflight_identity_for",
    "reset_preflight_identity",
    "resolve_selected_gpu",
]


if __name__ == "__main__":
    raise SystemExit(main())
