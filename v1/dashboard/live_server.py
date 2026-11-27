"""Thread-safe real-time SUMO + TraCI HTTP JSON bridge for zero-blink Canvas."""

from __future__ import annotations

from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time
from typing import Any

from dashboard.canvas_viz import VIEW_HALF_X_M, VIEW_HALF_Y_M
from dashboard.live_sim import LiveSimSession, SimSnapshot


SPEED_SLEEP_SECONDS: dict[str, float] = {
    "1x": 0.45,
    "2x": 0.22,
    "4x": 0.10,
}


def snapshot_to_json_dict(
    snap: SimSnapshot, *, is_running: bool, speed_label: str
) -> dict[str, Any]:
    """Serialize SimSnapshot into a lightweight JSON-ready dict for Canvas."""
    prev_map = {v["id"]: v for v in snap.prev_vehicles}
    vehs = []
    for v in snap.vehicles:
        if abs(v["x"]) > VIEW_HALF_X_M + 10.0 or abs(v["y"]) > VIEW_HALF_Y_M + 10.0:
            continue
        pv = prev_map.get(v["id"])
        vehs.append(
            {
                "id": v["id"],
                "x0": pv["x"] if pv is not None else v["x"],
                "y0": pv["y"] if pv is not None else v["y"],
                "a0": (
                    pv.get("angle", v.get("angle", 0.0))
                    if pv is not None
                    else v.get("angle", 0.0)
                ),
                "x1": v["x"],
                "y1": v["y"],
                "a1": v.get("angle", 0.0),
                "vtype": v["vtype"],
                "lane_id": v.get("lane_id", ""),
                "speed": v["speed"],
            }
        )

    data = asdict(snap)
    data["vehicles"] = vehs
    data.pop("prev_vehicles", None)
    data["is_running"] = is_running
    data["speed_label"] = speed_label
    data["poll_ms"] = int(SPEED_SLEEP_SECONDS.get(speed_label, 0.45) * 1000)
    return data


class LiveSimBridge:
    """Runs LiveSimSession in a background thread and serves /state over localhost."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._session: LiveSimSession | None = None
        self._speed_label: str = "1x"
        self._stop_event = threading.Event()
        self._latest_json: str = "{}"

        self._server = self._start_http_server()
        self._worker = threading.Thread(
            target=self._run_loop, name="LiveSimBridgeWorker", daemon=True
        )
        self._worker.start()

    @property
    def state_url(self) -> str:
        port = self._server.server_address[1]
        return f"http://127.0.0.1:{port}/state"

    def _start_http_server(self) -> ThreadingHTTPServer:
        bridge = self

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path.split("?")[0] != "/state":
                    self.send_response(404)
                    self.end_headers()
                    return
                body = bridge.get_latest_json().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return

        return ThreadingHTTPServer(("127.0.0.1", 0), _Handler)

    def _serve_http(self) -> None:
        self._server.serve_forever()

    def ensure_session(self, scenario: str, controller_key: str) -> None:
        with self._lock:
            if (
                self._session is None
                or self._session.scenario != scenario
                or self._session.controller_key != controller_key
            ):
                if self._session is not None:
                    self._session.close()
                self._session = LiveSimSession(
                    scenario=scenario, controller_key=controller_key
                )
                snap = self._session.reset(warmup_steps=20)
                self._latest_json = json.dumps(
                    snapshot_to_json_dict(
                        snap,
                        is_running=self._session.is_running,
                        speed_label=self._speed_label,
                    )
                )

    def start(self) -> None:
        with self._lock:
            if self._session is not None and not self._session.terminated:
                self._session.is_running = True
                self._refresh_json_locked()

    def pause(self) -> None:
        with self._lock:
            if self._session is not None:
                self._session.is_running = False
                self._refresh_json_locked()

    def reset(self, warmup_steps: int = 20) -> None:
        with self._lock:
            if self._session is not None:
                self._session.is_running = False
                snap = self._session.reset(warmup_steps=warmup_steps)
                self._latest_json = json.dumps(
                    snapshot_to_json_dict(
                        snap, is_running=False, speed_label=self._speed_label
                    )
                )

    def set_speed(self, speed_label: str) -> None:
        with self._lock:
            if speed_label in SPEED_SLEEP_SECONDS:
                self._speed_label = speed_label
                self._refresh_json_locked()

    def get_snapshot(self) -> SimSnapshot | None:
        with self._lock:
            return self._session.snapshot if self._session is not None else None

    def get_status(self) -> tuple[bool, bool]:
        with self._lock:
            if self._session is None:
                return False, False
            return self._session.is_running, self._session.terminated

    def get_latest_json(self) -> str:
        with self._lock:
            return self._latest_json

    def _refresh_json_locked(self) -> None:
        if self._session is not None and self._session.snapshot is not None:
            self._latest_json = json.dumps(
                snapshot_to_json_dict(
                    self._session.snapshot,
                    is_running=self._session.is_running,
                    speed_label=self._speed_label,
                )
            )

    def _run_loop(self) -> None:
        http_thread = threading.Thread(
            target=self._serve_http, name="LiveSimBridgeHTTP", daemon=True
        )
        http_thread.start()

        while not self._stop_event.is_set():
            sleep_s = 0.25
            with self._lock:
                if (
                    self._session is not None
                    and self._session.is_running
                    and not self._session.terminated
                ):
                    snap = self._session.step_once()
                    self._latest_json = json.dumps(
                        snapshot_to_json_dict(
                            snap,
                            is_running=self._session.is_running,
                            speed_label=self._speed_label,
                        )
                    )
                    sleep_s = SPEED_SLEEP_SECONDS.get(self._speed_label, 0.45)
            time.sleep(sleep_s)

    def stop(self) -> None:
        self._stop_event.set()
        with self._lock:
            if self._session is not None:
                self._session.close()
                self._session = None
        try:
            self._server.shutdown()
            self._server.server_close()
        except Exception:  # noqa: BLE001
            pass
