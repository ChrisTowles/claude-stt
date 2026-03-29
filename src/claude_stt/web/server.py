"""HTTP + WebSocket server for Chrome Web Speech API bridge."""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import threading
from pathlib import Path
from typing import Callable, Optional

from aiohttp import web

_logger = logging.getLogger(__name__)
_HTML_PATH = Path(__file__).parent / "index.html"


class WebSpeechServer:
    """Serves HTML page and WebSocket for Chrome Web Speech API communication."""

    def __init__(
        self,
        port: int = 18333,
        on_interim: Optional[Callable[[str], None]] = None,
        on_final: Optional[Callable[[str], None]] = None,
        on_end: Optional[Callable[[], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
        on_ready: Optional[Callable[[], None]] = None,
    ):
        self.port = port
        self.on_interim = on_interim
        self.on_final = on_final
        self.on_end = on_end
        self.on_error = on_error
        self.on_ready = on_ready

        # Random session token — only the Chrome window launched by this daemon can connect
        self._session = secrets.token_urlsafe(8)
        self._ws: Optional[web.WebSocketResponse] = None
        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/{self._session}"

    def is_connected(self) -> bool:
        return self._ws is not None and not self._ws.closed

    def start(self) -> bool:
        try:
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(
                target=self._run_loop, name="claude-stt-web", daemon=True
            )
            self._thread.start()
            import time
            time.sleep(0.5)
            return True
        except Exception:
            _logger.exception("Failed to start web server")
            return False

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._start_server())
        self._loop.run_forever()

    async def _start_server(self):
        self._app = web.Application()
        self._app.router.add_get(f"/{self._session}", self._handle_index)
        self._app.router.add_get(f"/{self._session}/ws", self._handle_ws)

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", self.port)
        await site.start()
        _logger.info("Web server listening at %s", self.url)

    async def _handle_index(self, request: web.Request) -> web.Response:
        html = _HTML_PATH.read_text()
        return web.Response(text=html, content_type="text/html")

    async def _handle_ws(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=30.0)
        await ws.prepare(request)

        # Close any existing connection (only one client allowed)
        old_ws = self._ws
        if old_ws and not old_ws.closed:
            _logger.info("Replacing existing WebSocket connection")
            await old_ws.close()
        self._ws = ws
        _logger.info("Chrome WebSocket connected")

        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    self._handle_message(msg.data)
                elif msg.type in (web.WSMsgType.ERROR, web.WSMsgType.CLOSE, web.WSMsgType.CLOSING):
                    break
        except Exception:
            _logger.exception("WebSocket handler error")
        finally:
            if self._ws is ws:
                self._ws = None
            _logger.info("Chrome WebSocket disconnected")

        return ws

    def _handle_message(self, data: str):
        try:
            msg = json.loads(data)
        except json.JSONDecodeError:
            _logger.warning("Invalid WebSocket message: %s", data)
            return

        msg_type = msg.get("type")
        if msg_type == "interim" and self.on_interim:
            self.on_interim(msg.get("text", ""))
        elif msg_type == "final" and self.on_final:
            self.on_final(msg.get("text", ""))
        elif msg_type == "end" and self.on_end:
            self.on_end()
        elif msg_type == "error" and self.on_error:
            self.on_error(msg.get("error", "unknown"))
        elif msg_type == "ready" and self.on_ready:
            self.on_ready()

    def send_start(self):
        self._send({"type": "start"})

    def send_stop(self):
        self._send({"type": "stop"})

    def _send(self, msg: dict):
        if not self._loop or not self.is_connected():
            return
        asyncio.run_coroutine_threadsafe(self._async_send(msg), self._loop)

    async def _async_send(self, msg: dict):
        if self._ws and not self._ws.closed:
            try:
                await self._ws.send_json(msg)
            except Exception:
                _logger.debug("Failed to send WebSocket message", exc_info=True)

    def stop(self):
        if self._loop:
            asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop)
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread:
                self._thread.join(timeout=5.0)

    async def _shutdown(self):
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._runner:
            await self._runner.cleanup()
