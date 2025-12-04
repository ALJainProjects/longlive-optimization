#!/usr/bin/env python
# LongLive Interactive Demo Server
# SPDX-License-Identifier: Apache-2.0
"""
FastAPI server for interactive LongLive video generation with WebRTC streaming.

Usage:
    python demo/server.py --config configs/longlive_optimized.yaml --port 8000
"""

import argparse
import asyncio
import json
import time
from typing import Optional
import uuid

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn

try:
    from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
    from aiortc.contrib.media import MediaRelay
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False
    print("Warning: aiortc not installed. WebRTC features disabled.")

from demo.pipeline_wrapper import OptimizedLongLivePipeline


app = FastAPI(title="LongLive Interactive Demo")

# Global state
pipeline: Optional[OptimizedLongLivePipeline] = None
pcs = set()  # Active peer connections


class VideoGeneratorTrack(VideoStreamTrack if WEBRTC_AVAILABLE else object):
    """
    Video track that generates frames from LongLive pipeline.
    """

    def __init__(self, pipeline: OptimizedLongLivePipeline):
        if WEBRTC_AVAILABLE:
            super().__init__()
        self.pipeline = pipeline
        self.frame_count = 0
        self.start_time = time.time()

    async def recv(self):
        if not WEBRTC_AVAILABLE:
            raise RuntimeError("WebRTC not available")

        from av import VideoFrame
        import numpy as np

        # Get next frame from pipeline
        frame_data = await self.pipeline.get_next_frame()

        if frame_data is None:
            # Generate placeholder frame
            frame_data = np.zeros((480, 832, 3), dtype=np.uint8)

        # Create video frame
        frame = VideoFrame.from_ndarray(frame_data, format="rgb24")
        frame.pts = self.frame_count
        frame.time_base = self.pipeline.time_base
        self.frame_count += 1

        return frame


@app.on_event("startup")
async def startup():
    """Initialize pipeline on startup."""
    global pipeline

    # Load config (passed via environment or default)
    import os
    config_path = os.environ.get("LONGLIVE_CONFIG", "configs/longlive_optimized.yaml")

    print(f"Loading pipeline from {config_path}...")
    pipeline = OptimizedLongLivePipeline(config_path)
    await pipeline.initialize()
    print("Pipeline ready!")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    global pipeline

    # Close all peer connections
    for pc in pcs:
        await pc.close()
    pcs.clear()

    if pipeline:
        await pipeline.cleanup()


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the demo UI."""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>LongLive Interactive Demo</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #1a1a1a; color: white; }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { text-align: center; }
        .video-container { text-align: center; margin: 20px 0; }
        video { max-width: 100%; border: 2px solid #333; border-radius: 8px; }
        .controls { display: flex; gap: 10px; margin: 20px 0; }
        .controls input { flex: 1; padding: 10px; font-size: 16px; border: none; border-radius: 4px; }
        .controls button { padding: 10px 20px; font-size: 16px; cursor: pointer; border: none; border-radius: 4px; background: #007acc; color: white; }
        .controls button:hover { background: #005a9e; }
        .metrics { display: flex; gap: 20px; justify-content: center; margin: 20px 0; }
        .metric { text-align: center; padding: 10px 20px; background: #333; border-radius: 8px; }
        .metric-value { font-size: 24px; font-weight: bold; color: #00ff88; }
        .metric-label { font-size: 12px; color: #888; }
        .status { text-align: center; padding: 10px; background: #333; border-radius: 4px; margin: 10px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>LongLive Interactive Demo</h1>

        <div class="video-container">
            <video id="video" autoplay playsinline></video>
        </div>

        <div class="controls">
            <input type="text" id="prompt" placeholder="Enter your prompt..." value="A beautiful sunset over the ocean">
            <button onclick="updatePrompt()">Generate</button>
            <button onclick="toggleStream()">Start/Stop</button>
        </div>

        <div class="metrics">
            <div class="metric">
                <div class="metric-value" id="fps">0.0</div>
                <div class="metric-label">FPS</div>
            </div>
            <div class="metric">
                <div class="metric-value" id="latency">0</div>
                <div class="metric-label">Latency (ms)</div>
            </div>
            <div class="metric">
                <div class="metric-value" id="frames">0</div>
                <div class="metric-label">Frames</div>
            </div>
        </div>

        <div class="status" id="status">Ready to connect</div>
    </div>

    <script>
        let ws = null;
        let pc = null;
        let streaming = false;

        function setStatus(msg) {
            document.getElementById('status').textContent = msg;
        }

        function updateMetrics(data) {
            if (data.fps) document.getElementById('fps').textContent = data.fps.toFixed(1);
            if (data.latency_ms) document.getElementById('latency').textContent = Math.round(data.latency_ms);
            if (data.frame_count) document.getElementById('frames').textContent = data.frame_count;
        }

        async function connect() {
            setStatus('Connecting...');

            // WebSocket for signaling and control
            ws = new WebSocket(`ws://${window.location.host}/ws`);

            ws.onopen = () => {
                setStatus('WebSocket connected');
            };

            ws.onmessage = async (event) => {
                const msg = JSON.parse(event.data);

                if (msg.type === 'offer') {
                    // Create peer connection
                    pc = new RTCPeerConnection({
                        iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
                    });

                    pc.ontrack = (event) => {
                        document.getElementById('video').srcObject = event.streams[0];
                        setStatus('Streaming');
                    };

                    pc.onicecandidate = (event) => {
                        if (event.candidate) {
                            ws.send(JSON.stringify({
                                type: 'ice-candidate',
                                candidate: event.candidate
                            }));
                        }
                    };

                    // Set remote description
                    await pc.setRemoteDescription(new RTCSessionDescription(msg.sdp));

                    // Create answer
                    const answer = await pc.createAnswer();
                    await pc.setLocalDescription(answer);

                    ws.send(JSON.stringify({
                        type: 'answer',
                        sdp: pc.localDescription
                    }));
                }
                else if (msg.type === 'metrics') {
                    updateMetrics(msg);
                }
            };

            ws.onclose = () => {
                setStatus('Disconnected');
                streaming = false;
            };
        }

        function updatePrompt() {
            if (ws && ws.readyState === WebSocket.OPEN) {
                const prompt = document.getElementById('prompt').value;
                ws.send(JSON.stringify({
                    type: 'prompt_update',
                    prompt: prompt
                }));
                setStatus('Prompt updated: ' + prompt);
            }
        }

        async function toggleStream() {
            if (!streaming) {
                await connect();
                if (ws) {
                    ws.send(JSON.stringify({ type: 'start' }));
                }
                streaming = true;
            } else {
                if (ws) {
                    ws.send(JSON.stringify({ type: 'stop' }));
                    ws.close();
                }
                if (pc) {
                    pc.close();
                }
                streaming = false;
                setStatus('Stopped');
            }
        }

        // Connect on page load
        // connect();
    </script>
</body>
</html>
    """


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for signaling and control."""
    await websocket.accept()

    session_id = str(uuid.uuid4())
    print(f"New WebSocket connection: {session_id}")

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)

            if msg["type"] == "start":
                # Start streaming
                if pipeline:
                    await pipeline.start_generation()
                    # Send offer (simplified - real implementation needs proper WebRTC)
                    await websocket.send_json({
                        "type": "status",
                        "message": "Generation started"
                    })

            elif msg["type"] == "stop":
                if pipeline:
                    await pipeline.stop_generation()

            elif msg["type"] == "prompt_update":
                if pipeline:
                    prompt = msg.get("prompt", "")
                    await pipeline.update_prompt(prompt)
                    await websocket.send_json({
                        "type": "prompt_applied",
                        "prompt": prompt,
                        "timestamp": time.time()
                    })

            # Send metrics periodically
            if pipeline:
                metrics = pipeline.get_metrics()
                await websocket.send_json({
                    "type": "metrics",
                    **metrics
                })

    except WebSocketDisconnect:
        print(f"WebSocket disconnected: {session_id}")


@app.get("/api/status")
async def get_status():
    """Get current pipeline status."""
    if pipeline is None:
        return {"status": "not_loaded"}

    return {
        "status": "ready" if pipeline.is_ready else "loading",
        "generating": pipeline.is_generating,
        "current_prompt": pipeline.current_prompt,
        "metrics": pipeline.get_metrics()
    }


@app.post("/api/prompt")
async def set_prompt(prompt: str):
    """Update the generation prompt."""
    if pipeline:
        await pipeline.update_prompt(prompt)
        return {"status": "updated", "prompt": prompt}
    return {"status": "error", "message": "Pipeline not loaded"}


def main():
    parser = argparse.ArgumentParser(description="LongLive Interactive Demo Server")
    parser.add_argument("--config", type=str, default="configs/longlive_optimized.yaml")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import os
    os.environ["LONGLIVE_CONFIG"] = args.config

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
