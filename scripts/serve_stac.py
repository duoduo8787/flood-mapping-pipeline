"""
serve_stac.py — 帶 CORS headers 的靜態檔案伺服器
用法：python3 serve_stac.py
"""
from http.server import SimpleHTTPRequestHandler, HTTPServer

class CORSHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, format, *args):
        pass  # 關掉 log 減少雜訊

if __name__ == "__main__":
    port = 8888
    print(f"Serving http://localhost:{port}/ with CORS enabled")
    HTTPServer(("", port), CORSHandler).serve_forever()
