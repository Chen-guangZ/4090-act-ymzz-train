"""Serve the Isaac Sim render over a loopback HTTP port."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


PAGE = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>K3 Isaac Sim 实时画面</title>
<style>
body{margin:0;background:#111419;color:#f5f7fa;font:16px system-ui}
main{margin:0 auto;max-width:1360px;padding:20px}
h1{font-size:22px;margin:0 0 6px}p{color:#bbc5d3;margin:0 0 14px}img{display:block;max-width:100%;max-height:80vh}
</style>
<main><h1>K3 #21 · Isaac Sim</h1><p id="status">正在连接 4090 的只读画面…</p>
<img id="view" alt="Isaac Sim 渲染的 K3 #21 回放"></main>
<script>
const view=document.getElementById("view"),status=document.getElementById("status");
const next=()=>{view.src="/frame.jpg?"+Date.now()};
view.onload=()=>{status.textContent="已连接 · 约 37 秒一轮，画面自动循环";setTimeout(next,330)};
view.onerror=()=>{status.textContent="暂时无法取得画面，正在重试";setTimeout(next,1200)};
next();
</script>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frame", type=Path)
    parser.add_argument("--port", type=int, default=18765)
    args = parser.parse_args()

    class ViewHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.partition("?")[0]
            if path == "/":
                self.send_bytes(PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/frame.jpg":
                try:
                    data = args.frame.read_bytes()
                except OSError:
                    self.send_error(503, "The Isaac Sim frame is unavailable")
                    return
                self.send_bytes(data, "image/jpeg")
            else:
                self.send_error(404)

        def send_bytes(self, body, mime):
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", args.port), ViewHandler) as server:
        print(f"Viewing Isaac Sim renderer on 127.0.0.1:{args.port}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
