"""
Minimal local HTTP server for the Hand Tracking Art experience.
Serves files from the current directory on port 8080.

Usage:
    python3 server.py

Then open: http://localhost:8080
"""
import http.server
import socketserver

PORT = 8080

Handler = http.server.SimpleHTTPRequestHandler
Handler.extensions_map.update({'.js': 'application/javascript'})

with socketserver.TCPServer(('', PORT), Handler) as httpd:
    print(f'Serving on http://localhost:{PORT}')
    httpd.serve_forever()
