# Review: fusion_tools/upload_logs.py

## 1. Command injection via `shell=True` - severity: critical
`cmd = f"scp {path} {dest_host}:/uploads/"` is executed with `subprocess.call(cmd, shell=True)`.
Both `run_id` and `dest_host` come from the HTTP request, so `dest_host="x; rm -rf ~"` executes
arbitrary commands. Fix: pass an argv list without a shell
(`subprocess.run(["scp", "--", path, f"{host}:/uploads/"], check=True)`) and validate `dest_host`
against an allowlist of servers.

## 2. Path traversal in `run_id` - severity: high
`os.path.join(LOG_DIR, run_id)` with `run_id="../../etc/shadow"` (or an absolute path) reads and
uploads arbitrary files. Fix: validate `run_id` with a strict pattern and check that the resolved
path stays inside `LOG_DIR`.

## 3. Hardcoded production credential - severity: high
`API_TOKEN` is committed in source. Rotate it and load it from a secret store / environment.

## 4. TLS verification disabled (`verify=False`) - severity: high
Allows man-in-the-middle interception of the bearer token. Remove `verify=False`.

## 5. Swallowed exception leads to NameError - severity: medium
`except Exception: pass` hides malformed metadata and then `meta["frames"]` raises
`NameError`/`UnboundLocalError`. Catch `json.JSONDecodeError`/`IndexError` explicitly and fail
with a clear error.

## 6. Exit status of scp ignored - severity: medium
`subprocess.call` return code is ignored, so failed uploads are registered as successful. Use
`check=True` (and a timeout).

## 7. Unbounded read and leaked file handle - severity: low
`open(path).read()` loads arbitrarily large logs into memory and never closes the file. Use a
context manager and read only the first line / enforce a size limit. Also `requests.post` has no
timeout.

## Verdict
Request changes - critical security issues must be fixed before merge.
