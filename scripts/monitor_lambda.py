#!/usr/bin/env python3
"""
Lambda Labs Instance & Job Monitor

Monitors:
- Instance status, uptime, and cost
- Job progress via SSH
- GPU utilization and memory
- Benchmark progress

Usage:
    python scripts/monitor_lambda.py --instance-id <id>
    python scripts/monitor_lambda.py --instance-ip <ip>
    python scripts/monitor_lambda.py --auto  # Find running instances

Features:
    --watch         Continuous monitoring (refreshes every 10s)
    --slack-webhook Send alerts to Slack
    --cost-limit    Alert when cost exceeds limit
"""

import os
import sys
import json
import time
import argparse
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
import threading

# Try to import optional dependencies
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


@dataclass
class InstanceInfo:
    """Lambda Labs instance information."""
    instance_id: str
    name: str
    status: str
    ip: Optional[str]
    instance_type: str
    region: str
    hostname: str
    jupyter_token: Optional[str]
    jupyter_url: Optional[str]
    ssh_key_names: List[str]
    created_at: Optional[datetime] = None
    hourly_cost: float = 2.49  # Default H100 cost

    @property
    def uptime_seconds(self) -> float:
        if self.created_at:
            return (datetime.now() - self.created_at).total_seconds()
        return 0

    @property
    def uptime_str(self) -> str:
        seconds = self.uptime_seconds
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"

    @property
    def estimated_cost(self) -> float:
        return (self.uptime_seconds / 3600) * self.hourly_cost


@dataclass
class JobProgress:
    """Job progress tracking."""
    stage: str = "unknown"
    progress_pct: float = 0.0
    current_config: str = ""
    configs_completed: int = 0
    total_configs: int = 9
    fps: float = 0.0
    eta_minutes: float = 0.0
    last_log_line: str = ""
    gpu_util: float = 0.0
    gpu_memory_used: float = 0.0
    gpu_memory_total: float = 0.0


@dataclass
class MonitorState:
    """Overall monitor state."""
    instance: Optional[InstanceInfo] = None
    job: JobProgress = field(default_factory=JobProgress)
    alerts: List[str] = field(default_factory=list)
    last_update: Optional[datetime] = None


class LambdaLabsAPI:
    """Lambda Labs API client."""

    BASE_URL = "https://cloud.lambdalabs.com/api/v1"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session() if HAS_REQUESTS else None
        if self.session:
            self.session.auth = (api_key, "")

    def _request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """Make API request."""
        if not HAS_REQUESTS:
            # Fallback to curl
            return self._curl_request(method, endpoint, **kwargs)

        url = f"{self.BASE_URL}/{endpoint}"
        resp = self.session.request(method, url, **kwargs)
        resp.raise_for_status()
        return resp.json()

    def _curl_request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """Fallback to curl for API requests."""
        url = f"{self.BASE_URL}/{endpoint}"
        cmd = [
            "curl", "-s", "-u", f"{self.api_key}:",
            "-X", method.upper(), url
        ]
        if kwargs.get("json"):
            cmd.extend(["-H", "Content-Type: application/json"])
            cmd.extend(["-d", json.dumps(kwargs["json"])])

        result = subprocess.run(cmd, capture_output=True, text=True)
        return json.loads(result.stdout)

    def list_instances(self) -> List[InstanceInfo]:
        """List all instances."""
        data = self._request("GET", "instances")
        instances = []
        for inst_data in data.get("data", []):
            instances.append(self._parse_instance(inst_data))
        return instances

    def get_instance(self, instance_id: str) -> Optional[InstanceInfo]:
        """Get specific instance."""
        try:
            data = self._request("GET", f"instances/{instance_id}")
            return self._parse_instance(data.get("data", {}))
        except Exception:
            return None

    def _parse_instance(self, data: Dict) -> InstanceInfo:
        """Parse instance data."""
        # Try to parse creation time from instance data
        created_at = None
        # Lambda API doesn't provide created_at directly, estimate from uptime

        return InstanceInfo(
            instance_id=data.get("id", ""),
            name=data.get("name", ""),
            status=data.get("status", "unknown"),
            ip=data.get("ip"),
            instance_type=data.get("instance_type", {}).get("name", "unknown"),
            region=data.get("region", {}).get("name", "unknown"),
            hostname=data.get("hostname", ""),
            jupyter_token=data.get("jupyter_token"),
            jupyter_url=data.get("jupyter_url"),
            ssh_key_names=data.get("ssh_key_names", []),
            created_at=created_at,
        )

    def get_instance_types(self) -> Dict:
        """Get available instance types with pricing."""
        return self._request("GET", "instance-types")


class SSHJobMonitor:
    """Monitor job progress via SSH."""

    # Default SSH key path for Lambda Labs instances
    DEFAULT_SSH_KEY = os.path.expanduser("~/.ssh/lambda_gh200")

    def __init__(self, host: str, user: str = "ubuntu", ssh_key: Optional[str] = None):
        self.host = host
        self.user = user
        # Use provided key, or default, or try without key
        self.ssh_key = ssh_key or (self.DEFAULT_SSH_KEY if os.path.exists(self.DEFAULT_SSH_KEY) else None)

    def _ssh_cmd(self, command: str, timeout: int = 10) -> Optional[str]:
        """Execute SSH command."""
        try:
            ssh_args = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5",
                        "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR"]
            # Add key file if available
            if self.ssh_key and os.path.exists(self.ssh_key):
                ssh_args.extend(["-i", self.ssh_key])
            ssh_args.append(f"{self.user}@{self.host}")
            ssh_args.append(command)

            result = subprocess.run(
                ssh_args,
                capture_output=True, text=True, timeout=timeout
            )
            return result.stdout.strip()
        except (subprocess.TimeoutExpired, subprocess.SubprocessError):
            return None

    def get_gpu_stats(self) -> Dict[str, float]:
        """Get GPU utilization and memory."""
        output = self._ssh_cmd(
            "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total "
            "--format=csv,noheader,nounits 2>/dev/null"
        )
        if not output:
            return {}

        try:
            parts = output.split(",")
            return {
                "gpu_util": float(parts[0].strip()),
                "memory_used": float(parts[1].strip()) / 1024,  # GB
                "memory_total": float(parts[2].strip()) / 1024,  # GB
            }
        except (IndexError, ValueError):
            return {}

    def get_job_stage(self) -> str:
        """Detect current job stage."""
        # Check for running processes
        processes = self._ssh_cmd("ps aux | grep -E 'python|setup_h100' | grep -v grep")

        if not processes:
            return "idle"

        if "setup_h100" in processes:
            return "setup"
        if "run_benchmark" in processes or "run_full_benchmark" in processes:
            return "benchmarking"
        if "collect_quality" in processes:
            return "quality_metrics"
        if "generate_comparison" in processes:
            return "generating_videos"
        if "python" in processes:
            return "running_python"

        return "unknown"

    def get_benchmark_progress(self) -> Dict[str, Any]:
        """Parse benchmark progress from logs."""
        # Check for benchmark output files
        output = self._ssh_cmd(
            "ls -la ~/longlive-optimization/benchmark_results/*.json 2>/dev/null | wc -l"
        )
        configs_done = int(output) if output and output.isdigit() else 0

        # Get last log lines
        log_tail = self._ssh_cmd(
            "tail -5 ~/longlive-optimization/benchmark.log 2>/dev/null || "
            "tail -5 /tmp/benchmark.log 2>/dev/null"
        )

        # Try to parse FPS from logs
        fps = 0.0
        if log_tail:
            for line in log_tail.split("\n"):
                if "FPS" in line:
                    try:
                        fps = float(line.split("FPS")[0].split()[-1])
                    except (IndexError, ValueError):
                        pass

        return {
            "configs_completed": configs_done,
            "total_configs": 9,
            "last_log": log_tail.split("\n")[-1] if log_tail else "",
            "fps": fps,
        }

    def get_disk_usage(self) -> Dict[str, str]:
        """Get disk usage."""
        output = self._ssh_cmd("df -h / | tail -1")
        if output:
            parts = output.split()
            return {
                "total": parts[1] if len(parts) > 1 else "?",
                "used": parts[2] if len(parts) > 2 else "?",
                "avail": parts[3] if len(parts) > 3 else "?",
                "pct": parts[4] if len(parts) > 4 else "?",
            }
        return {}

    def get_job_progress_file(self) -> Optional[Dict]:
        """Read job progress JSON from remote instance."""
        output = self._ssh_cmd(
            "cat /tmp/benchmark_progress.json 2>/dev/null || "
            "cat ~/longlive-optimization/benchmark_results/progress.json 2>/dev/null"
        )
        if output:
            try:
                return json.loads(output)
            except json.JSONDecodeError:
                pass
        return None

    def get_live_benchmark_output(self, lines: int = 20) -> str:
        """Get live benchmark output from screen or tmux."""
        # Try to get output from various sources
        output = self._ssh_cmd(
            f"tail -{lines} ~/longlive-optimization/benchmark_results/*.log 2>/dev/null || "
            f"tail -{lines} /tmp/benchmark*.log 2>/dev/null || "
            "pgrep -a python | head -5"
        )
        return output or ""


class SlackNotifier:
    """Send alerts to Slack."""

    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    def send(self, message: str, color: str = "#36a64f"):
        """Send Slack message."""
        if not HAS_REQUESTS:
            return

        payload = {
            "attachments": [{
                "color": color,
                "text": message,
                "footer": "Lambda Labs Monitor",
                "ts": int(time.time())
            }]
        }
        try:
            requests.post(self.webhook_url, json=payload, timeout=5)
        except Exception:
            pass


class LambdaMonitor:
    """Main monitoring class."""

    def __init__(
        self,
        api_key: str,
        instance_id: Optional[str] = None,
        instance_ip: Optional[str] = None,
        slack_webhook: Optional[str] = None,
        cost_limit: Optional[float] = None,
        ssh_key: Optional[str] = None,
    ):
        self.api = LambdaLabsAPI(api_key)
        self.instance_id = instance_id
        self.instance_ip = instance_ip
        self.slack = SlackNotifier(slack_webhook) if slack_webhook else None
        self.cost_limit = cost_limit
        self.ssh_key = ssh_key
        self.state = MonitorState()
        self.start_time = datetime.now()
        self._cost_alert_sent = False

    def find_instance(self) -> Optional[InstanceInfo]:
        """Find instance by ID or IP, or get first running instance."""
        instances = self.api.list_instances()

        if self.instance_id:
            for inst in instances:
                if inst.instance_id == self.instance_id:
                    return inst

        if self.instance_ip:
            for inst in instances:
                if inst.ip == self.instance_ip:
                    return inst

        # Return first active instance
        for inst in instances:
            if inst.status == "active":
                return inst

        return None

    def update(self) -> MonitorState:
        """Update monitor state."""
        # Update instance info
        self.state.instance = self.find_instance()
        self.state.last_update = datetime.now()

        if not self.state.instance:
            return self.state

        # Set start time if not set
        if not self.state.instance.created_at:
            self.state.instance.created_at = self.start_time

        # Update job progress if we have an IP
        if self.state.instance.ip:
            ssh = SSHJobMonitor(self.state.instance.ip)

            # Get GPU stats
            gpu = ssh.get_gpu_stats()
            self.state.job.gpu_util = gpu.get("gpu_util", 0)
            self.state.job.gpu_memory_used = gpu.get("memory_used", 0)
            self.state.job.gpu_memory_total = gpu.get("memory_total", 0)

            # Get job stage
            self.state.job.stage = ssh.get_job_stage()

            # Get benchmark progress
            progress = ssh.get_benchmark_progress()
            self.state.job.configs_completed = progress.get("configs_completed", 0)
            self.state.job.total_configs = progress.get("total_configs", 9)
            self.state.job.fps = progress.get("fps", 0)
            self.state.job.last_log_line = progress.get("last_log", "")

            # Calculate progress percentage
            if self.state.job.stage == "setup":
                self.state.job.progress_pct = 10.0
            elif self.state.job.stage == "benchmarking":
                self.state.job.progress_pct = 10 + (
                    80 * self.state.job.configs_completed / self.state.job.total_configs
                )
            elif self.state.job.stage == "quality_metrics":
                self.state.job.progress_pct = 90.0
            elif self.state.job.stage == "generating_videos":
                self.state.job.progress_pct = 95.0
            elif self.state.job.stage == "idle":
                if self.state.job.configs_completed > 0:
                    self.state.job.progress_pct = 100.0

        # Check alerts
        self._check_alerts()

        return self.state

    def _check_alerts(self):
        """Check for alert conditions."""
        if not self.state.instance:
            return

        # Cost limit alert
        if self.cost_limit and not self._cost_alert_sent:
            if self.state.instance.estimated_cost >= self.cost_limit:
                msg = (f"⚠️ Cost limit reached!\n"
                       f"Current: ${self.state.instance.estimated_cost:.2f}\n"
                       f"Limit: ${self.cost_limit:.2f}")
                self.state.alerts.append(msg)
                if self.slack:
                    self.slack.send(msg, color="#ff0000")
                self._cost_alert_sent = True

        # Job completion alert
        if self.state.job.progress_pct >= 100:
            if "Job complete" not in str(self.state.alerts):
                msg = f"✅ Job complete on {self.state.instance.instance_id}"
                self.state.alerts.append(msg)
                if self.slack:
                    self.slack.send(msg, color="#36a64f")

    def format_status(self) -> str:
        """Format status for display."""
        lines = []
        lines.append("=" * 60)
        lines.append("LAMBDA LABS MONITOR")
        lines.append("=" * 60)

        if not self.state.instance:
            lines.append("❌ No instance found")
            return "\n".join(lines)

        inst = self.state.instance
        job = self.state.job

        # Instance info
        lines.append(f"\n📦 INSTANCE")
        lines.append(f"   ID:     {inst.instance_id}")
        lines.append(f"   Type:   {inst.instance_type}")
        lines.append(f"   Region: {inst.region}")
        lines.append(f"   Status: {inst.status}")
        lines.append(f"   IP:     {inst.ip or 'pending'}")

        # Timing & Cost
        lines.append(f"\n⏱️  TIME & COST")
        lines.append(f"   Uptime:  {inst.uptime_str}")
        lines.append(f"   Rate:    ${inst.hourly_cost:.2f}/hr")
        lines.append(f"   Cost:    ${inst.estimated_cost:.2f}")
        if self.cost_limit:
            lines.append(f"   Limit:   ${self.cost_limit:.2f}")

        # GPU stats
        if job.gpu_memory_total > 0:
            lines.append(f"\n🎮 GPU")
            lines.append(f"   Util:    {job.gpu_util:.0f}%")
            lines.append(f"   Memory:  {job.gpu_memory_used:.1f}/{job.gpu_memory_total:.1f} GB")

        # Job progress
        lines.append(f"\n📊 JOB PROGRESS")
        lines.append(f"   Stage:    {job.stage}")
        lines.append(f"   Progress: {job.progress_pct:.0f}%")

        # Progress bar
        bar_width = 40
        filled = int(bar_width * job.progress_pct / 100)
        bar = "█" * filled + "░" * (bar_width - filled)
        lines.append(f"   [{bar}]")

        if job.stage == "benchmarking":
            lines.append(f"   Configs:  {job.configs_completed}/{job.total_configs}")
            if job.fps > 0:
                lines.append(f"   FPS:      {job.fps:.1f}")

        if job.last_log_line:
            lines.append(f"\n📝 LAST LOG")
            lines.append(f"   {job.last_log_line[:60]}")

        # Alerts
        if self.state.alerts:
            lines.append(f"\n🚨 ALERTS")
            for alert in self.state.alerts[-3:]:
                lines.append(f"   {alert}")

        lines.append("")
        lines.append(f"Last update: {self.state.last_update.strftime('%H:%M:%S')}")
        lines.append("=" * 60)

        return "\n".join(lines)


def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def main():
    parser = argparse.ArgumentParser(description="Lambda Labs Instance & Job Monitor")
    parser.add_argument("--instance-id", type=str, help="Instance ID to monitor")
    parser.add_argument("--instance-ip", type=str, help="Instance IP to monitor")
    parser.add_argument("--auto", action="store_true", help="Auto-detect running instance")
    parser.add_argument("--watch", action="store_true", default=True, help="Continuous monitoring (default: True)")
    parser.add_argument("--no-watch", action="store_true", help="Single status check (disable continuous)")
    parser.add_argument("--interval", type=int, default=10, help="Refresh interval (seconds)")
    parser.add_argument("--slack-webhook", type=str, help="Slack webhook URL for alerts")
    parser.add_argument("--cost-limit", type=float, help="Cost limit for alerts ($)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    # Load API key
    api_key = os.environ.get("LAMBDA_API_KEY")
    if not api_key:
        creds_file = Path.home() / ".ray-cluster-credentials"
        if creds_file.exists():
            with open(creds_file) as f:
                for line in f:
                    if "LAMBDA_API_KEY" in line:
                        api_key = line.split("=")[1].strip().strip('"')
                        break

    if not api_key:
        print("ERROR: LAMBDA_API_KEY not found")
        print("Set it with: export LAMBDA_API_KEY=your_key")
        print("Or add to ~/.ray-cluster-credentials")
        sys.exit(1)

    # Create monitor
    monitor = LambdaMonitor(
        api_key=api_key,
        instance_id=args.instance_id,
        instance_ip=args.instance_ip,
        slack_webhook=args.slack_webhook,
        cost_limit=args.cost_limit,
    )

    try:
        # Continuous monitoring is default; use --no-watch for single check
        watch_mode = args.watch and not args.no_watch
        if watch_mode:
            while True:
                clear_screen()
                monitor.update()

                if args.json:
                    print(json.dumps({
                        "instance": {
                            "id": monitor.state.instance.instance_id if monitor.state.instance else None,
                            "status": monitor.state.instance.status if monitor.state.instance else None,
                            "ip": monitor.state.instance.ip if monitor.state.instance else None,
                            "uptime": monitor.state.instance.uptime_str if monitor.state.instance else None,
                            "cost": monitor.state.instance.estimated_cost if monitor.state.instance else 0,
                        },
                        "job": {
                            "stage": monitor.state.job.stage,
                            "progress": monitor.state.job.progress_pct,
                            "gpu_util": monitor.state.job.gpu_util,
                        },
                        "alerts": monitor.state.alerts,
                    }, indent=2))
                else:
                    print(monitor.format_status())
                    print(f"\nRefreshing every {args.interval}s... (Ctrl+C to stop)")

                time.sleep(args.interval)
        else:
            monitor.update()
            if args.json:
                print(json.dumps({
                    "instance": {
                        "id": monitor.state.instance.instance_id if monitor.state.instance else None,
                        "status": monitor.state.instance.status if monitor.state.instance else None,
                        "ip": monitor.state.instance.ip if monitor.state.instance else None,
                    },
                    "job": {
                        "stage": monitor.state.job.stage,
                        "progress": monitor.state.job.progress_pct,
                    },
                }, indent=2))
            else:
                print(monitor.format_status())

    except KeyboardInterrupt:
        print("\nMonitoring stopped.")


if __name__ == "__main__":
    main()
