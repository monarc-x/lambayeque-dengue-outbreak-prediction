"""Serve the local decision-support dashboard at http://127.0.0.1:8080."""

from __future__ import annotations

import http.server
import socketserver
import webbrowser
from pathlib import Path

WEB = Path(__file__).resolve().parent / "web"
PORT = 8080


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB), **kwargs)


if __name__ == "__main__":
    with socketserver.TCPServer(("127.0.0.1", PORT), Handler) as httpd:
        url = f"http://127.0.0.1:{PORT}/"
        print(f"Dashboard available at: {url}")
        print("Press Ctrl+C to stop")
        try:
            webbrowser.open(url)
        except Exception:
            pass
        httpd.serve_forever()
