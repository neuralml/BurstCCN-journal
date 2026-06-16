from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess


WANDB_DISABLED_ENV = {
    "WANDB_MODE": "disabled",
    "WANDB_SILENT": "true",
    "WANDB_CONSOLE": "off",
    "PYTHONUNBUFFERED": "1",
    "PYTHONIOENCODING": "utf-8",
    "PYTHONUTF8": "1",
}


def run_command(cmd, cwd):
    env = os.environ.copy()
    env.update(WANDB_DISABLED_ENV)
    completed = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output = f"{completed.stdout}\n{completed.stderr}"
    if completed.returncode != 0:
        log_dir = Path(cwd) / "multi_run_experiments" / "failed_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"failed_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.log"
        log_path.write_text(
            f"returncode={completed.returncode}\n"
            f"command={' '.join(cmd)}\n\n"
            f"{output}",
            encoding="utf-8",
        )
        output_tail = "\n".join(output.splitlines()[-200:])
        raise RuntimeError(
            f"Command failed with returncode={completed.returncode}\n"
            f"Command: {' '.join(cmd)}\n\n"
            f"Full output written to: {log_path}\n\n"
            f"{output_tail}"
        )
    return completed.returncode, output


def run_jobs(jobs, run_job, sort_results, save_results, n_parallel):
    results = []
    with ThreadPoolExecutor(max_workers=n_parallel) as executor:
        futures = {executor.submit(run_job, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            result = future.result()
            results.append(result)
            results = sort_results(results)
            save_results(results)
            yield job, result, results


def write_json_atomic(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)
