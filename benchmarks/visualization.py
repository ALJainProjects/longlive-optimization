# LongLive Benchmark Visualization
# SPDX-License-Identifier: Apache-2.0
"""
Time budget visualization for LongLive profiling results.
"""

import json
from typing import Dict, Any, List


def generate_time_budget_report(
    results: Dict[str, Any],
    output_path: str = None,
    format: str = "text"
) -> str:
    """
    Generate a time budget breakdown report.

    Args:
        results: Benchmark results dictionary
        output_path: Optional path to write output
        format: "text", "json", or "html"

    Returns:
        Report string
    """
    if format == "text":
        report = _generate_text_report(results)
    elif format == "json":
        report = _generate_json_report(results)
    elif format == "html":
        report = _generate_html_report(results)
    else:
        raise ValueError(f"Unknown format: {format}")

    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"Report saved to: {output_path}")

    return report


def _generate_text_report(results: Dict[str, Any]) -> str:
    """Generate ASCII time budget visualization."""
    lines = []
    lines.append("=" * 80)
    lines.append("LONGLIVE TIME BUDGET BREAKDOWN")
    lines.append("=" * 80)

    summary = results.get("summary", {})
    config = results.get("config", {})

    # Configuration
    lines.append(f"\nConfiguration:")
    lines.append(f"  Model: {config.get('model_name', 'N/A')}")
    lines.append(f"  Frames: {config.get('num_output_frames', 'N/A')}")
    lines.append(f"  Batch size: {config.get('batch_size', 'N/A')}")
    lines.append(f"  Local window: {config.get('local_attn_size', 'N/A')}")
    lines.append(f"  Denoising steps: {len(config.get('denoising_steps', []))}")
    lines.append(f"  GPU: {config.get('gpu_name', 'N/A')}")

    # Overall performance
    lines.append("\n" + "-" * 80)
    lines.append("PERFORMANCE SUMMARY")
    lines.append("-" * 80)

    fps = summary.get("fps", {})
    ms_frame = summary.get("ms_per_frame", {})

    lines.append(f"\n  FPS: {fps.get('mean', 0):.2f} +/- {fps.get('std', 0):.2f}")
    lines.append(f"  Latency per frame: {ms_frame.get('mean', 0):.2f} ms (p95: {ms_frame.get('p95', 0):.2f} ms)")
    lines.append(f"  Steady-state: {summary.get('steady_state_inter_frame_ms', {}).get('mean', 0):.2f} ms/frame")
    lines.append(f"  Peak memory: {summary.get('memory_peak_gb', {}).get('mean', 0):.2f} GB")

    # Expected breakdown based on paper
    total_ms = ms_frame.get('mean', 48.0)
    lines.append("\n" + "-" * 80)
    lines.append("ESTIMATED TIME BREAKDOWN (based on paper)")
    lines.append("-" * 80)

    breakdown = [
        ("Initialization", 0.06, "KV cache + cross-attn init"),
        ("Diffusion Loop", 0.81, "4 timesteps x 30 blocks"),
        ("  - Attention", 0.50, "Flash attention kernels"),
        ("  - FFN", 0.20, "Feed-forward layers"),
        ("  - KV Cache", 0.11, "Cache updates/rolling"),
        ("VAE Decode", 0.10, "Latent to pixel"),
        ("Other", 0.03, "Sync, overhead"),
    ]

    for name, pct, desc in breakdown:
        est_ms = total_ms * pct
        bar_len = int(pct * 50)
        bar = "#" * bar_len + "." * (50 - bar_len)
        lines.append(f"\n  {name}")
        lines.append(f"    [{bar}] {est_ms:.2f} ms ({pct*100:.1f}%)")
        lines.append(f"    {desc}")

    # Target assessment
    lines.append("\n" + "-" * 80)
    lines.append("TARGET ASSESSMENT (40ms for real-time)")
    lines.append("-" * 80)

    target = 40.0
    mean_ms = ms_frame.get('mean', 0)

    if mean_ms <= target:
        lines.append(f"\n  [PASS] Mean latency ({mean_ms:.2f} ms) meets target ({target} ms)")
        margin = target - mean_ms
        lines.append(f"         Margin: {margin:.2f} ms ({margin/target*100:.1f}%)")
    else:
        lines.append(f"\n  [FAIL] Mean latency ({mean_ms:.2f} ms) exceeds target ({target} ms)")
        gap = mean_ms - target
        reduction_needed = (gap / mean_ms) * 100
        lines.append(f"         Gap: {gap:.2f} ms")
        lines.append(f"         Reduction needed: {reduction_needed:.1f}%")

    lines.append("\n" + "=" * 80)

    return "\n".join(lines)


def _generate_json_report(results: Dict[str, Any]) -> str:
    """Generate JSON report."""
    report = {
        "summary": results.get("summary", {}),
        "config": results.get("config", {}),
        "time_budget": {
            "initialization_pct": 0.06,
            "diffusion_pct": 0.81,
            "vae_decode_pct": 0.10,
            "other_pct": 0.03,
        },
        "target_assessment": {
            "target_ms": 40.0,
            "mean_ms": results.get("summary", {}).get("ms_per_frame", {}).get("mean", 0),
            "meets_target": results.get("summary", {}).get("ms_per_frame", {}).get("mean", 0) <= 40.0,
        }
    }
    return json.dumps(report, indent=2)


def _generate_html_report(results: Dict[str, Any]) -> str:
    """Generate interactive HTML visualization."""
    summary = results.get("summary", {})
    config = results.get("config", {})

    ms_frame = summary.get("ms_per_frame", {}).get("mean", 48.0)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>LongLive Time Budget</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .card {{ background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; border-bottom: 2px solid #007acc; padding-bottom: 10px; }}
        .chart {{ width: 100%; height: 400px; }}
        .metric {{ display: inline-block; margin: 10px 20px; text-align: center; }}
        .metric-value {{ font-size: 36px; font-weight: bold; color: #007acc; }}
        .metric-label {{ font-size: 14px; color: #666; }}
        .pass {{ color: #28a745; }}
        .fail {{ color: #dc3545; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>LongLive Time Budget Analysis</h1>

        <div class="card">
            <h2>Key Metrics</h2>
            <div class="metric">
                <div class="metric-value">{summary.get('fps', {}).get('mean', 0):.1f}</div>
                <div class="metric-label">FPS</div>
            </div>
            <div class="metric">
                <div class="metric-value">{ms_frame:.1f}</div>
                <div class="metric-label">ms/frame</div>
            </div>
            <div class="metric">
                <div class="metric-value">{summary.get('memory_peak_gb', {}).get('mean', 0):.1f}</div>
                <div class="metric-label">Peak GB</div>
            </div>
            <div class="metric">
                <div class="metric-value {'pass' if ms_frame <= 40 else 'fail'}">
                    {'PASS' if ms_frame <= 40 else 'FAIL'}
                </div>
                <div class="metric-label">Target (40ms)</div>
            </div>
        </div>

        <div class="card">
            <h2>Time Budget Breakdown</h2>
            <div id="pie-chart" class="chart"></div>
        </div>

        <div class="card">
            <h2>Latency Distribution</h2>
            <div id="box-chart" class="chart"></div>
        </div>

        <div class="card">
            <h2>Configuration</h2>
            <table>
                <tr><td><b>Model:</b></td><td>{config.get('model_name', 'N/A')}</td></tr>
                <tr><td><b>Frames:</b></td><td>{config.get('num_output_frames', 'N/A')}</td></tr>
                <tr><td><b>Window size:</b></td><td>{config.get('local_attn_size', 'N/A')}</td></tr>
                <tr><td><b>GPU:</b></td><td>{config.get('gpu_name', 'N/A')}</td></tr>
                <tr><td><b>torch.compile:</b></td><td>{config.get('use_torch_compile', False)}</td></tr>
                <tr><td><b>FP8:</b></td><td>{config.get('use_fp8', False)}</td></tr>
            </table>
        </div>
    </div>

    <script>
        // Pie chart for time budget
        var pieData = [{{
            values: [{ms_frame * 0.06:.2f}, {ms_frame * 0.81:.2f}, {ms_frame * 0.10:.2f}, {ms_frame * 0.03:.2f}],
            labels: ['Initialization', 'Diffusion', 'VAE Decode', 'Other'],
            type: 'pie',
            hole: 0.4,
            textinfo: 'label+percent',
            marker: {{
                colors: ['#4CAF50', '#2196F3', '#FF9800', '#9E9E9E']
            }}
        }}];

        Plotly.newPlot('pie-chart', pieData, {{
            title: 'Time Budget Breakdown (Estimated)',
            showlegend: true
        }});

        // Box plot for latency
        var boxData = [{{
            y: [{ms_frame:.2f}],
            type: 'box',
            name: 'Latency',
            marker: {{ color: '#2196F3' }}
        }}];

        Plotly.newPlot('box-chart', boxData, {{
            title: 'Frame Latency Distribution',
            yaxis: {{ title: 'Latency (ms)' }},
            shapes: [{{
                type: 'line',
                x0: -0.5,
                x1: 0.5,
                y0: 40,
                y1: 40,
                line: {{ color: 'red', width: 2, dash: 'dash' }}
            }}]
        }});
    </script>
</body>
</html>"""

    return html


def generate_optimization_comparison_chart(results_list: List[Dict], output_path: str):
    """Generate comparison chart for multiple optimization configs."""
    configs = [r.get("config_name", f"config_{i}") for i, r in enumerate(results_list)]
    fps_values = [r.get("fps_mean", 0) for r in results_list]
    latency_values = [r.get("ms_per_frame_mean", 0) for r in results_list]
    memory_values = [r.get("memory_peak_gb", 0) for r in results_list]

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>LongLive Optimization Comparison</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .chart {{ width: 100%; height: 400px; margin: 20px 0; }}
    </style>
</head>
<body>
    <h1>Optimization Comparison</h1>

    <div id="fps-chart" class="chart"></div>
    <div id="latency-chart" class="chart"></div>

    <script>
        var configs = {json.dumps(configs)};
        var fpsValues = {json.dumps(fps_values)};
        var latencyValues = {json.dumps(latency_values)};

        Plotly.newPlot('fps-chart', [{{
            x: configs,
            y: fpsValues,
            type: 'bar',
            marker: {{ color: '#2196F3' }}
        }}], {{
            title: 'FPS by Configuration',
            yaxis: {{ title: 'FPS' }}
        }});

        Plotly.newPlot('latency-chart', [{{
            x: configs,
            y: latencyValues,
            type: 'bar',
            marker: {{ color: '#FF9800' }}
        }}], {{
            title: 'Latency by Configuration',
            yaxis: {{ title: 'ms/frame' }},
            shapes: [{{
                type: 'line',
                x0: -0.5,
                x1: configs.length - 0.5,
                y0: 40,
                y1: 40,
                line: {{ color: 'red', width: 2, dash: 'dash' }}
            }}]
        }});
    </script>
</body>
</html>"""

    with open(output_path, 'w') as f:
        f.write(html)

    print(f"Comparison chart saved to: {output_path}")
