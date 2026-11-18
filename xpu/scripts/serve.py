"""
Launch vLLM on Intel XPU from a model config (models/<id>/model.yaml).

  python3 scripts/serve.py models/qwen3.8-27b-exl3-4.00bpw --gpu 0            # one replica
  python3 scripts/serve.py models/qwen3.8-27b-exl3-4.00bpw --dp                # replica per GPU + LB
  python3 scripts/serve.py models/... --gpu 0 --print                           # show the command only
  python3 scripts/serve.py models/... --gpu 0 -- --enforce-eager                # extra vllm args

--model-path overrides where the weights are (default: $MODELS_DIR/<repo name> or the HF repo id).

The API listens on 127.0.0.1 unless --host (or $SERVE_HOST) says otherwise. Any other address needs an API key,
read from --api-key-file (or the file named by $SERVE_API_KEY_FILE); clients then send "Authorization: Bearer <key>".
--no-api-key serves another address without a key, for a container whose port is published only to the host's
loopback (docker run -p 127.0.0.1:8000:8000). In --dp mode the replicas always listen on 127.0.0.1 behind lb.py.
"""
from __future__ import annotations
import argparse, ipaddress, json, os, shlex, signal, subprocess, sys, time
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOOPBACK = "127.0.0.1"
# Set through serve.py's own options so the address and key checks below always apply.
RESERVED = ("--host", "--api-key")


def load(path):
    if os.path.isdir(path):
        path = os.path.join(path, "model.yaml")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_dir"] = os.path.dirname(os.path.abspath(path))
    return cfg


def model_path(cfg, override):
    if override:
        return override
    repo = cfg["source"]["repo"]
    d = os.environ.get("MODELS_DIR")
    if d:
        for cand in (cfg["id"], repo.split("/")[-1], repo.replace("/", "-")):
            p = os.path.join(d, cand)
            if os.path.isdir(p):
                return p
    return repo


def is_loopback(host):
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False                  # a hostname may resolve anywhere


def read_api_key(path):
    if not path:
        return None
    with open(path) as f:
        key = f.read().strip()
    if not key:
        sys.exit(f"serve.py: API key file {path} is empty")
    return key


def vllm_argv(cfg, path, host, port, api_key, extra):
    v = cfg["vllm"]
    argv = ["vllm", "serve", path, "--served-model-name", cfg["served_model_name"],
            "--host", host, "--port", str(port)]
    if api_key:
        argv += ["--api-key", api_key]
    if not os.path.isdir(path):
        argv += ["--revision", cfg["source"]["revision"]]
    for k, val in v.items():
        flag = "--" + k.replace("_", "-")
        if val is None:
            continue                  # null in the config / --set k=null drops the flag
        if isinstance(val, bool):
            if val:
                argv.append(flag)
        elif isinstance(val, (dict, list)):
            argv += [flag, json.dumps(val, separators=(",", ":"))]
        else:
            argv += [flag, str(val)]
    return argv + list(extra)


def shown(argv):
    """The command as printed: the API key is never echoed."""
    return shlex.join("<api key>" if i and argv[i - 1] == "--api-key" else a for i, a in enumerate(argv))


def env_for(cfg, gpu):
    env = dict(os.environ)
    mdir = cfg.get("_dir", "")
    for k, v in (cfg.get("env") or {}).items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = str(v).replace("{model_dir}", mdir)
    env["ZE_AFFINITY_MASK"] = str(gpu)
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--port", type=int)
    ap.add_argument("--dp", action="store_true", help="one replica per GPU in parallel.gpus + load balancer")
    ap.add_argument("--model-path")
    ap.add_argument("--host", default=os.environ.get("SERVE_HOST") or LOOPBACK,
                    help="address the API listens on (default $SERVE_HOST or 127.0.0.1); in --dp mode, lb.py's address")
    ap.add_argument("--api-key-file", default=os.environ.get("SERVE_API_KEY_FILE") or None,
                    help="file holding the API key clients must send (default: the file named by $SERVE_API_KEY_FILE)")
    ap.add_argument("--no-api-key", action="store_true",
                    help="serve a non-loopback --host without a key (only when the port is published to loopback)")
    ap.add_argument("--print", action="store_true")
    ap.add_argument("--set", action="append", default=[], metavar="KEY.PATH=VALUE",
                    help="override a config value, e.g. vllm.speculative_config.num_speculative_tokens=5 (YAML value)")
    ap.add_argument("--log-dir", default=os.path.join(ROOT, "logs"))
    args, extra = ap.parse_known_args()
    extra = [e for e in extra if e != "--"]
    for e in extra:
        if e.split("=", 1)[0] in RESERVED:
            sys.exit(f"serve.py: pass {e.split('=', 1)[0]} as serve.py's own option (--host, --api-key-file), "
                     "not after --")
    api_key = read_api_key(args.api_key_file)
    if not api_key and not args.no_api_key and not is_loopback(args.host):
        sys.exit(f"serve.py: refusing to serve {args.host} without an API key. Set --api-key-file (or "
                 "$SERVE_API_KEY_FILE), or pass --no-api-key if the port is published only to the host's loopback.")
    cfg = load(args.config)
    for item in args.set:
        key, _, val = item.partition("=")
        node = cfg
        *parents, leaf = key.split(".")
        for k in parents:
            node = node.setdefault(k, {})
        try:
            node[leaf] = yaml.safe_load(val)
        except yaml.YAMLError:
            node[leaf] = val          # e.g. "{model_dir}/x.json" is not valid YAML; keep the raw string
    path = model_path(cfg, args.model_path)

    if not args.dp:
        port = args.port or cfg.get("parallel", {}).get("base_port", 8100)
        argv = vllm_argv(cfg, path, args.host, port, api_key, extra)
        if args.print:
            print(f"ZE_AFFINITY_MASK={args.gpu} " + shown(argv))
            return
        os.execvpe(argv[0], argv, env_for(cfg, args.gpu))

    par = cfg["parallel"]
    os.makedirs(args.log_dir, exist_ok=True)
    procs, backends = [], []
    for i, gpu in enumerate(par["gpus"]):
        port = par["base_port"] + i
        # Replicas stay on loopback; lb.py is the only listener on --host. With a key, each replica checks it
        # (lb.py forwards the Authorization header).
        argv = vllm_argv(cfg, path, LOOPBACK, port, api_key, extra)
        if args.print:
            print(f"ZE_AFFINITY_MASK={gpu} " + shown(argv))
            continue
        log = open(os.path.join(args.log_dir, f"serve{gpu}.log"), "w")
        procs.append(subprocess.Popen(argv, env=env_for(cfg, gpu), stdout=log, stderr=subprocess.STDOUT))
        backends.append(f"http://{LOOPBACK}:{port}")
    lb = [sys.executable, os.path.join(ROOT, "scripts", "lb.py"), "--host", args.host, "--port", str(par["lb_port"]),
          "--backends", ",".join(backends or [f"http://{LOOPBACK}:{par['base_port'] + i}" for i in range(len(par["gpus"]))])]
    if args.print:
        print(shlex.join(lb))
        return
    procs.append(subprocess.Popen(lb, stdout=open(os.path.join(args.log_dir, "lb.log"), "w"), stderr=subprocess.STDOUT))

    def stop(*_, code=0):
        for p in procs:
            if p.poll() is None:
                p.send_signal(signal.SIGTERM)
        sys.exit(code)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while all(p.poll() is None for p in procs):
        time.sleep(2)
    dead = [i for i, p in enumerate(procs) if p.poll() is not None]
    print(f"serve.py: process(es) {dead} exited; see {args.log_dir}", file=sys.stderr)
    stop(code=1)


if __name__ == "__main__":
    main()
