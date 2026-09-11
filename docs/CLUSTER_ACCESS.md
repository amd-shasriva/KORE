# Reaching the cluster, and getting the weights off it

Written after losing a compute allocation and discovering the data was still
there. Everything below was verified against the live cluster rather than
recalled, on `crsuse2-slog-005` in September 2026.

If you only want the model, skip to [section 3](#3-retrieving-the-weights).

## 0. The thing that is not obvious

**Losing your GPU node does not lose your files.** Compute and storage are
separate systems on SPUR. When a reservation ends or a node is decommissioned,
the *node* goes away; `/shared_nfs` stays mounted on the login nodes, and
anything you left there is still readable.

This is worth internalising before you panic, because the intuition from a
single-machine workflow is exactly backwards: the node you trained on being
unreachable says nothing about whether your checkpoints survive.

The corollary is the part that bites. Files on `/shared_nfs` outlive your
*allocation*, but they live under a personal directory, so they do not
necessarily outlive your *account*. Those are different deadlines.

## 1. Getting in

```bash
ssh <your_NTID>@crs-spur.crusoe.amd.com
```

Authenticate with your NTID and LDAP password. `crs-spur` is a load balancer
across three login nodes; address them directly if it misbehaves:

```
crsuse2-slog-003.crusoe.amd.com
crsuse2-slog-004.crusoe.amd.com
crsuse2-slog-005.crusoe.amd.com
```

Two traps here, both of which cost hours on this project.

**The compute nodes are a different machine family.** `crs-m2m-cpu-spur-*` is
not the same as `crsuse2-slog-*`, and their home directories are not the same
filesystem. A key added to `~/.ssh/authorized_keys` on one is *not* authorised
on the other, which produces a `Permission denied (publickey)` that looks like
a key problem and is really a which-machine problem. `crs-m2m-cpu-spur-login`
was decommissioned on 2026-09-08.

**The login shell is tcsh**, specifically `/tool/pandora/bin/tcsh`. Any command
you send with bash syntax will fail in ways that read as nonsense —
`Ambiguous output redirect` for a `2>/dev/null`, `Badly placed ()'s` for a
heredoc. Wrap remote commands explicitly:

```bash
ssh <NTID>@crs-spur.crusoe.amd.com "bash -c 'your; commands; here'"
```

Use `bash -c`, **not** `bash -lc`. The login profile stalls on NFS: measured,
`bash -lc 'hostname'` hung past 60 seconds where `bash -c 'hostname'` returned
in under three. For anything with awkward quoting, base64 the script instead of
fighting two shells at once:

```bash
B=$(base64 -w0 script.sh)
ssh <NTID>@crs-spur.crusoe.amd.com "echo $B | base64 -d | bash"
```

## 2. Where things live

| Mount | Size | Used | Use it for |
| --- | --- | --- | --- |
| `/shared_nfs` | 360 TB | 93% | Everything large. Visible from every node. |
| `/home` | 10 TB | **99%** | Code and configs only. Do not stage a checkpoint here. |

`/home` being at 99% is not a rounding error, it is 182 GB free on a shared
volume. A 114 GB copy into `$HOME` may succeed and leave the filesystem full for
everyone else. Stage large artifacts on `/shared_nfs`.

Available on the login nodes, verified: `rsync`, `scp`, `tar`, `sha256sum`,
`python3`, plus `sinfo`, `squeue` and `spur`. Do not run heavy compute here;
that is what an allocation is for.

## 3. Retrieving the weights

RL checkpoint-30 is at `/shared_nfs/shasriva/rl_checkpoint-30`, 342 GB, and
every file in it is world-readable. Start by reading what is next to it:

```bash
cat /shared_nfs/shasriva/KORE_HANDOFF/README.md
```

### The one command

From a clone of this repository:

```bash
./scripts/fetch_weights.sh ~/kore-weights            # 114 GB, the model
./scripts/fetch_weights.sh ~/kore-weights --resume   # 342 GB, adds training state
```

Set `KORE_SPUR_SSH=<NTID>@crs-spur.crusoe.amd.com` if you are running it from
somewhere other than a login node. It resumes a broken transfer, then verifies
the shard count and reads `global_step` back.

### By hand

Identical effect, if you would rather see the mechanics:

```bash
# weights only: what you want to evaluate, serve, or fine-tune
rsync -av --partial --append-verify \
  --exclude optimizer.pt --exclude rng_state.pth --exclude scheduler.pt \
  /shared_nfs/shasriva/rl_checkpoint-30/ ~/kore-weights/

# or everything, only if you intend to resume the interrupted run
rsync -av --partial --append-verify \
  /shared_nfs/shasriva/rl_checkpoint-30/ ~/kore-weights-full/
```

`--partial --append-verify` matters at this size. Without it an interrupted
114 GB transfer starts over; with it, rsync continues and re-checksums the
blocks it reuses rather than trusting them. Pulling to a different machine, put
the remote on the left:

```bash
rsync -av --partial --append-verify \
  --exclude optimizer.pt --exclude rng_state.pth --exclude scheduler.pt \
  <NTID>@crs-spur.crusoe.amd.com:/shared_nfs/shasriva/rl_checkpoint-30/ \
  ~/kore-weights/
```

### Which of the two you want

**Take the 114 GB.** The extra 228 GB is `optimizer.pt`, `rng_state.pth` and
`scheduler.pt`, needed only to *continue* training from step 30. Evaluating,
serving and fine-tuning all work without them.

Do not gzip this. Safetensors are already dense binary; compression buys
approximately nothing and costs hours of CPU. `rsync` or plain `tar` is correct.

### Verifying what you got

```bash
ls ~/kore-weights/model-*-of-00025.safetensors | wc -l   # 25
python3 -c "import json;print(json.load(open('$HOME/kore-weights/trainer_state.json'))['global_step'])"   # 30
```

Check `global_step`, not the directory name: two checkpoint directories exist
from different cycles, and the name is not evidence. `config.json` should read
`qwen3_moe` with 48 layers and 128 experts, which is
`Qwen/Qwen3-Coder-30B-A3B-Instruct` at revision
`b2cff646eb4bb1d68355c01b18ae02e7cf42d120`.

`/shared_nfs/shasriva/KORE_HANDOFF/CHECKPOINT_MANIFEST.txt` lists every file
with its exact byte size, so a transfer can be checked rather than assumed.

Then load it:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(
    "~/kore-weights", torch_dtype="bfloat16", device_map="auto")
tok = AutoTokenizer.from_pretrained("~/kore-weights")
```

## 4. The permission trap, if you are the one leaving files behind

This nearly cost this project its weights, and it is invisible until the moment
it matters.

The safetensors shards were written `-rw-------` — owner-read only — while the
directories containing them were world-readable `drwxrwxr-x`. From outside,
everything looks fine: a colleague can list the directory, see 25 shards of
plausible size, and start an rsync. It fails on the first file, with a
permission error, at a point where the owner may no longer have an account to
fix it with.

Check the **files**, not the directory:

```bash
find /shared_nfs/<your_dir> -type f ! -perm -o+r | wc -l    # anything but 0 is a problem
```

Fix it before you lose access:

```bash
chmod -R a+rX /shared_nfs/<your_dir>
```

`a+rX` rather than `a+rx`: the capital `X` adds execute only to directories and
to files that already had it, so it does not mark every data file executable.

## 5. When it does not work

| Symptom | Cause | Fix |
| --- | --- | --- |
| `Permission denied (publickey,password)` | Key is not in `authorized_keys` on *that* machine family | Add it on the login node, or use your LDAP password |
| `Connection refused` | Wrong port, or a tunnel that is not running | Check the host and port; `ss -ltn` locally |
| `kex_exchange_identification: Connection closed` | Nothing listening at the far end, or you tripped a rate limit | Confirm the host is up; wait before retrying |
| Repeated closes after many attempts | sshd rate limiting from too many connections | Stop, wait a few minutes; retrying faster makes it worse |
| `Ambiguous output redirect`, `Badly placed ()'s` | Your bash syntax hit tcsh | Wrap in `bash -c '...'` |
| Command hangs ~60s with no output | `bash -lc` sourcing a profile that stalls on NFS | Use `bash -c` |
| Host key mismatch for a tunnel alias | Tunnel now terminates on a different node | `ssh-keygen -R <alias>` and reconnect |

If you are reaching the cluster through an SSH tunnel, remember the tunnel only
gives you *reachability*. It carries no credential: the far-side sshd still
demands a key it recognises or a password, and a working tunnel with an
unauthorised key looks exactly like a broken tunnel unless you read the error.

## 6. If the directory is gone

`/shared_nfs/shasriva/` belongs to an account that is closing. If it has been
removed, the recipe (`configs/grpo_coder30b_a3b_trloo_frontier.json`) and the
whole training corpus are committed to this repository, so the work is
reproducible — but no stage is deterministic (`docs/REPRODUCING.md` section 8),
so a rerun yields a different checkpoint and every published figure would need
re-measuring rather than re-confirming. Ask cluster operations who owns that
path now.

See [`../HANDOVER.md`](../HANDOVER.md) for the full artifact inventory.
