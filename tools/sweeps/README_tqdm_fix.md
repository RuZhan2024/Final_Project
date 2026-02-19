# Sweep logs + tqdm fix

This bundle upgrades `tools/sweeps/sweep_lib.py` to run each `make train-*` under a pseudo-TTY on POSIX.
That makes tqdm progress bars render normally (single in-place bar) instead of printing a new line per update.

Install from repo root:

```bash
unzip -o sweep_upgraded_pty_logs.zip -d .
rm -rf tools/sweeps/__pycache__
rm -f tools/sweeps/sweep_lib*.pyc
```

Then run your sweeps as usual.
