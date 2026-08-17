"""Check the Startup shortcut is written, read back, repaired, and removed.

Everything runs against a temp directory standing in for the real Startup
folder, so running this never changes what your machine does at login.

Covers: install writes a shortcut the shell can read back, is_current() spots a
shortcut left behind by a moved venv, and sync() adds and deletes to match the
setting. Not covered here (needs a reboot): that Windows actually runs it.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from win32com.client import Dispatch  # noqa: E402

from localflow import autostart  # noqa: E402

failures = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global failures
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        failures += 1


tmpdir = Path(tempfile.mkdtemp(prefix="localflow-startup-"))
autostart.startup_dir = lambda: tmpdir

link_path = tmpdir / autostart.SHORTCUT_NAME
exe, args, workdir = autostart._target()

check("nothing installed to begin with", not autostart.is_current())
check("removing what is not there is not an error", autostart.remove() is False)

written = autostart.install()
check("install writes the shortcut", written.exists(), f"({written.name})")

link = Dispatch("WScript.Shell").CreateShortcut(str(link_path))
check("target is this interpreter", link.TargetPath.lower() == exe.lower(), f"({link.TargetPath})")
check("arguments run the package", link.Arguments.strip() == "-m localflow", f"({link.Arguments})")
check("working directory is the project", link.WorkingDirectory.lower() == workdir.lower())
check("window style is minimised", link.WindowStyle == 7, f"({link.WindowStyle})")
check("is_current sees its own work", autostart.is_current())

# A shortcut left behind by a venv that has since been rebuilt elsewhere. This
# is the case sync() has to repair; left alone it silently starts nothing.
stale = Dispatch("WScript.Shell").CreateShortcut(str(link_path))
stale.TargetPath = str(Path(workdir) / "gone" / "python.exe")
stale.Arguments = "-m localflow"
stale.WorkingDirectory = workdir
stale.Save()
check("stale target is detected", not autostart.is_current())
autostart.sync(True)
check("sync repairs a stale shortcut", autostart.is_current())

status = autostart.sync(True)
check("sync is idempotent", link_path.exists() and status == "starts with Windows", f"({status})")

status = autostart.sync(False)
check("sync removes when turned off", not link_path.exists(), f"({status})")
check("second removal reports nothing to do", autostart.sync(False) == "")

print(f"\n{'all autostart checks passed' if not failures else f'{failures} check(s) failed'}")
print(f"(temp startup folder: {tmpdir})")
raise SystemExit(1 if failures else 0)
