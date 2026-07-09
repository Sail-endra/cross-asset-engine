import http.server, socketserver, os
os.chdir(os.path.join(os.path.dirname(__file__), "..", "docs"))
socketserver.TCPServer(("", 8747), http.server.SimpleHTTPRequestHandler).serve_forever()
