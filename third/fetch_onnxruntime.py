"""Fetches the ONNX Runtime DirectML build into third/onnxruntime/.

The vcpkg onnxruntime port CANNOT build the DirectML execution provider: its
portfile maps `directml -> onnxruntime_USE_DML`, but no `directml` feature is
declared in its vcpkg.json, so that mapping can never activate. The only
alternatives are this official redist or a from-source build with --use_dml, so
the redist is vendored and this script is how it is reproduced.

The tree it writes is gitignored, like third/vcpkg_installed. Run it once after
cloning:

    python third/fetch_onnxruntime.py

DirectML.dll is deliberately NOT fetched. onnxruntime.dll imports it and
resolves against the in-box %WINDIR%\\System32\\DirectML.dll; the package
declares a dependency on Microsoft.AI.DirectML 1.15.4 and this script checks the
in-box copy is at least that. If it is not, install the Microsoft.AI.DirectML
redist alongside.
"""
import hashlib
import io
import os
import shutil
import sys
import urllib.request
import zipfile

VERSION = "1.24.4"
PACKAGE = "microsoft.ml.onnxruntime.directml"
URL = ("https://api.nuget.org/v3-flatcontainer/"
       f"{PACKAGE}/{VERSION}/{PACKAGE}.{VERSION}.nupkg")

# The DirectML version this ORT build was made against, from the package's
# nuspec dependency on Microsoft.AI.DirectML.
REQUIRED_DIRECTML = (1, 15, 4)

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "onnxruntime")


def in_box_directml_version():
    """(major, minor, patch) of the in-box DirectML.dll, or None."""
    path = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "System32", "DirectML.dll")
    if not os.path.exists(path):
        return None
    try:
        # win32api is not guaranteed; shell out to PowerShell instead.
        import subprocess
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             f"(Get-Item '{path}').VersionInfo.FileVersion"],
            capture_output=True, text=True, timeout=60).stdout.strip()
        # e.g. "1.15.5+250417-0100.1.os-germanium.7c460ce"
        head = out.split("+")[0]
        parts = [int(p) for p in head.split(".")[:3]]
        while len(parts) < 3:
            parts.append(0)
        return tuple(parts)
    except Exception:
        return None


def main():
    print(f"downloading {PACKAGE} {VERSION}")
    with urllib.request.urlopen(URL, timeout=300) as response:
        payload = response.read()
    print(f"  {len(payload)} bytes, sha256 {hashlib.sha256(payload).hexdigest()[:16]}...")

    if os.path.isdir(ROOT):
        shutil.rmtree(ROOT)

    written = 0
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for name in archive.namelist():
            base = os.path.basename(name)
            if name.startswith("build/native/include/") and name.endswith(".h"):
                dest = os.path.join(ROOT, "include", base)
            elif name.startswith("runtimes/win-x64/native/") and base.endswith(".lib"):
                dest = os.path.join(ROOT, "lib", base)
            elif name.startswith("runtimes/win-x64/native/") and base.endswith(".dll"):
                dest = os.path.join(ROOT, "bin", base)
            elif base.upper() in ("LICENSE", "LICENSE.TXT"):
                dest = os.path.join(ROOT, base)
            else:
                continue
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with archive.open(name) as src, open(dest, "wb") as out:
                out.write(src.read())
            written += 1

    print(f"  extracted {written} files to {ROOT}")

    version = in_box_directml_version()
    required = ".".join(str(p) for p in REQUIRED_DIRECTML)
    if version is None:
        print(f"  WARNING: could not read the in-box DirectML.dll version; "
              f"ONNX Runtime needs at least {required}")
    elif version < REQUIRED_DIRECTML:
        print(f"  WARNING: in-box DirectML is {'.'.join(str(p) for p in version)} but "
              f"ONNX Runtime {VERSION} needs {required}. Install the "
              f"Microsoft.AI.DirectML redist.")
        return 1
    else:
        print(f"  in-box DirectML {'.'.join(str(p) for p in version)} satisfies "
              f">= {required}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
