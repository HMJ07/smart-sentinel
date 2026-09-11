from flask import Flask, render_template_string, Response, jsonify
import sqlite3
import cv2

app = Flask(__name__)
frame_buffer = None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Smart Sentinel - Remote Dashboard</title>
    <style>
        body { font-family: monospace; background: #0f172a; color: #e2e8f0; margin: 20px; }
        h1 { color: #38bdf8; }
        .container { display: flex; gap: 20px; flex-wrap: wrap; }
        .video-box { border: 2px solid #334155; border-radius: 8px; overflow: hidden; }
        .events-box { flex-grow: 1; background: #1e293b; padding: 15px; border-radius: 8px; max-height: 500px; overflow-y: auto; }
        table { width: 100%; border-collapse: collapse; }
        th, td { text-align: left; padding: 8px; border-bottom: 1px solid #334155; }
        th { color: #38bdf8; }
    </style>
</head>
<body>
    <h1>🛡️ Smart Sentinel - Remote Dashboard</h1>
    <div class="container">
        <div class="video-box">
            <img src="/video_feed" width="640" height="360"/>
        </div>
        <div class="events-box">
            <h2>Historial de Eventos</h2>
            <table id="events-table">
                <thead>
                    <tr><th>ID</th><th>Timestamp</th><th>Tipo</th><th>Descripción</th></tr>
                </thead>
                <tbody></tbody>
            </table>
        </div>
    </div>
    <script>
        async function fetchEvents() {
            try {
                const res = await fetch('/api/events');
                const data = await res.json();
                const tbody = document.querySelector('#events-table tbody');
                tbody.innerHTML = data.map(e => "<tr><td>" + e.id + "</td><td>" + e.timestamp + "</td><td>" + e.event_type + "</td><td>" + e.description + "</td></tr>").join('');
            } catch(e) {}
        }
        setInterval(fetchEvents, 2000);
        fetchEvents();
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

def generate_feed():
    global frame_buffer
    while True:
        if frame_buffer is not None:
            _, buffer = cv2.imencode('.jpg', frame_buffer)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/events')
def get_events():
    try:
        with sqlite3.connect("events.db") as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT id, timestamp, event_type, description FROM events ORDER BY id DESC LIMIT 20")
            rows = cursor.fetchall()
            return jsonify([dict(r) for r in rows])
    except Exception:
        return jsonify([])

def run_dashboard():
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
