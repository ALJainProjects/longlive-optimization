"""
Job Callbacks for Progress Reporting

Provides callbacks for benchmark scripts to report progress to:
- Local log file
- Webhook endpoints (Slack, Discord, custom)
- File-based progress tracking

Usage:
    from utils.job_callbacks import JobReporter

    reporter = JobReporter(webhook_url="https://hooks.slack.com/...")

    reporter.start_job("benchmark")
    for i, config in enumerate(configs):
        reporter.update_progress(stage="benchmarking", progress=i/len(configs)*100)
        run_benchmark(config)
        reporter.log(f"Completed {config['name']}: {fps} FPS")
    reporter.complete_job(summary={"best_fps": 41.67})
"""

import os
import json
import time
import socket
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass, field, asdict
from threading import Thread
from queue import Queue

# Optional imports
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


@dataclass
class JobStatus:
    """Current job status."""
    job_id: str
    hostname: str
    stage: str = "initializing"
    progress_pct: float = 0.0
    current_task: str = ""
    tasks_completed: int = 0
    total_tasks: int = 0
    fps: float = 0.0
    latency_ms: float = 0.0
    memory_gb: float = 0.0
    start_time: str = ""
    elapsed_seconds: float = 0.0
    eta_seconds: float = 0.0
    logs: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


class WebhookSender:
    """Send updates to webhook endpoints."""

    def __init__(self, url: str, format: str = "slack"):
        self.url = url
        self.format = format

    def send(self, status: JobStatus, event: str = "update"):
        """Send status to webhook."""
        if not HAS_REQUESTS:
            return

        if self.format == "slack":
            payload = self._format_slack(status, event)
        elif self.format == "discord":
            payload = self._format_discord(status, event)
        else:
            payload = {"event": event, "status": status.to_dict()}

        try:
            requests.post(self.url, json=payload, timeout=5)
        except Exception as e:
            logging.warning(f"Webhook failed: {e}")

    def _format_slack(self, status: JobStatus, event: str) -> Dict:
        """Format for Slack."""
        if event == "start":
            color = "#36a64f"
            title = "🚀 Job Started"
        elif event == "complete":
            color = "#36a64f"
            title = "✅ Job Complete"
        elif event == "error":
            color = "#ff0000"
            title = "❌ Job Error"
        else:
            color = "#3498db"
            title = "📊 Job Update"

        # Progress bar
        bar_width = 20
        filled = int(bar_width * status.progress_pct / 100)
        bar = "█" * filled + "░" * (bar_width - filled)

        text = f"*{status.stage}*: {status.progress_pct:.0f}%\n"
        text += f"`[{bar}]`\n"

        if status.current_task:
            text += f"Task: {status.current_task}\n"
        if status.fps > 0:
            text += f"FPS: {status.fps:.1f} | Latency: {status.latency_ms:.1f}ms\n"

        elapsed = timedelta(seconds=int(status.elapsed_seconds))
        text += f"Elapsed: {elapsed}"

        if status.eta_seconds > 0:
            eta = timedelta(seconds=int(status.eta_seconds))
            text += f" | ETA: {eta}"

        return {
            "attachments": [{
                "color": color,
                "title": title,
                "text": text,
                "footer": f"{status.hostname} | {status.job_id}",
                "ts": int(time.time())
            }]
        }

    def _format_discord(self, status: JobStatus, event: str) -> Dict:
        """Format for Discord."""
        if event == "start":
            color = 3066993  # Green
        elif event == "complete":
            color = 3066993
        elif event == "error":
            color = 15158332  # Red
        else:
            color = 3447003  # Blue

        return {
            "embeds": [{
                "title": f"Job {event.title()}: {status.stage}",
                "color": color,
                "fields": [
                    {"name": "Progress", "value": f"{status.progress_pct:.0f}%", "inline": True},
                    {"name": "FPS", "value": f"{status.fps:.1f}", "inline": True},
                    {"name": "Task", "value": status.current_task or "-", "inline": False},
                ],
                "footer": {"text": f"{status.hostname}"}
            }]
        }


class FileProgressWriter:
    """Write progress to file for monitoring."""

    def __init__(self, path: str = "/tmp/job_progress.json"):
        self.path = Path(path)

    def write(self, status: JobStatus):
        """Write status to file."""
        self.path.write_text(json.dumps(status.to_dict(), indent=2))


class JobReporter:
    """
    Main job reporter with multiple backends.

    Usage:
        reporter = JobReporter(
            webhook_url="https://hooks.slack.com/...",
            progress_file="/tmp/progress.json"
        )

        reporter.start_job("benchmark")
        reporter.update_progress(stage="setup", progress=10)
        reporter.log("Installing dependencies...")
        reporter.update_progress(stage="benchmarking", progress=50, fps=35.5)
        reporter.complete_job(summary={"best_fps": 41.67})
    """

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        webhook_format: str = "slack",
        progress_file: str = "/tmp/job_progress.json",
        log_file: str = "/tmp/job.log",
        update_interval: float = 30.0,  # Min seconds between webhook updates
    ):
        self.job_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.hostname = socket.gethostname()

        self.status = JobStatus(
            job_id=self.job_id,
            hostname=self.hostname,
            start_time=datetime.now().isoformat()
        )

        self.webhook = WebhookSender(webhook_url, webhook_format) if webhook_url else None
        self.file_writer = FileProgressWriter(progress_file)
        self.log_path = Path(log_file)

        self.update_interval = update_interval
        self._last_webhook_time = 0
        self._start_time = time.time()

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("job")

    def start_job(self, name: str, total_tasks: int = 0):
        """Signal job start."""
        self.status.stage = name
        self.status.total_tasks = total_tasks
        self._start_time = time.time()

        self.logger.info(f"Starting job: {name}")
        self._send_update("start")

    def update_progress(
        self,
        stage: Optional[str] = None,
        progress: Optional[float] = None,
        task: Optional[str] = None,
        tasks_completed: Optional[int] = None,
        fps: Optional[float] = None,
        latency_ms: Optional[float] = None,
        memory_gb: Optional[float] = None,
        metrics: Optional[Dict] = None,
    ):
        """Update job progress."""
        if stage:
            self.status.stage = stage
        if progress is not None:
            self.status.progress_pct = progress
        if task:
            self.status.current_task = task
        if tasks_completed is not None:
            self.status.tasks_completed = tasks_completed
        if fps is not None:
            self.status.fps = fps
        if latency_ms is not None:
            self.status.latency_ms = latency_ms
        if memory_gb is not None:
            self.status.memory_gb = memory_gb
        if metrics:
            self.status.metrics.update(metrics)

        # Update timing
        self.status.elapsed_seconds = time.time() - self._start_time

        # Estimate ETA
        if self.status.progress_pct > 0:
            total_estimated = self.status.elapsed_seconds / (self.status.progress_pct / 100)
            self.status.eta_seconds = total_estimated - self.status.elapsed_seconds

        # Write to file (always)
        self.file_writer.write(self.status)

        # Send webhook (rate-limited)
        self._send_update("update", rate_limit=True)

    def log(self, message: str, level: str = "info"):
        """Log a message."""
        self.status.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
        self.status.logs = self.status.logs[-50:]  # Keep last 50

        getattr(self.logger, level)(message)

    def error(self, message: str):
        """Log an error."""
        self.status.errors.append(message)
        self.logger.error(message)
        self._send_update("error")

    def complete_job(self, summary: Optional[Dict] = None):
        """Signal job completion."""
        self.status.stage = "complete"
        self.status.progress_pct = 100.0
        self.status.elapsed_seconds = time.time() - self._start_time

        if summary:
            self.status.metrics.update(summary)

        self.logger.info(f"Job complete. Elapsed: {self.status.elapsed_seconds:.0f}s")
        self.file_writer.write(self.status)
        self._send_update("complete")

    def _send_update(self, event: str, rate_limit: bool = False):
        """Send update to webhook."""
        if not self.webhook:
            return

        if rate_limit:
            now = time.time()
            if now - self._last_webhook_time < self.update_interval:
                return
            self._last_webhook_time = now

        self.webhook.send(self.status, event)


# Convenience function for quick setup
def create_reporter(
    slack_webhook: Optional[str] = None,
    discord_webhook: Optional[str] = None,
) -> JobReporter:
    """Create a job reporter with common defaults."""
    # Check environment variables
    webhook_url = slack_webhook or os.environ.get("SLACK_WEBHOOK_URL")
    webhook_format = "slack"

    if not webhook_url:
        webhook_url = discord_webhook or os.environ.get("DISCORD_WEBHOOK_URL")
        webhook_format = "discord"

    return JobReporter(
        webhook_url=webhook_url,
        webhook_format=webhook_format,
    )


# Decorator for automatic progress tracking
def track_progress(reporter: JobReporter, stage: str):
    """Decorator to track function progress."""
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            reporter.update_progress(stage=stage, task=func.__name__)
            try:
                result = func(*args, **kwargs)
                reporter.log(f"Completed: {func.__name__}")
                return result
            except Exception as e:
                reporter.error(f"Failed: {func.__name__}: {e}")
                raise
        return wrapper
    return decorator


# Import timedelta for formatting
from datetime import timedelta
