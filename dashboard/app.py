import os
import sqlite3
import time

import cv2
from flask import Flask, Response, jsonify, render_template_string, send_from_directory

from config.settings import Config

app = Flask(__name__)
frame_buffer = None
status = {"level": "NORMAL", "summary": "Sin anomalías", "persons": 0, "acknowledged": False}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Smart Sentinel - Dashboard</title>
    <style>
        :root { --ok:#22c55e; --warn:#f59e0b; --crit:#ef4444; }
        body { font-family: ui-monospace, monospace; background: #0f172a; color: #e2e8f0; margin: 20px; }
        h1 { color: #38bdf8; margin-top: 0; }
        #status { padding: 14px 18px; border-radius: 8px; margin-bottom: 16px; font-weight: bold;
                  background: #1e293b; border-left: 8px solid var(--ok); }
        #status.WARNING { border-color: var(--warn); }
        #status.CRITICAL { border-color: var(--crit); animation: pulse 1s infinite; }
        #status small { display:block; font-weight: normal; color:#94a3b8; margin-top:4px; }
        @keyframes pulse { 50% { background: #3b1219; } }
        .container { display: flex; gap: 20px; flex-wrap: wrap; }
        .video-box { border: 2px solid #334155; border-radius: 8px; overflow: hidden; }
        .video-box img { width: 640px; max-width: 100%; height: auto; display:block; }
        .events-box { flex: 1; min-width: 320px; background: #1e293b; padding: 15px; border-radius: 8px; max-height: 520px; overflow-y: auto; }
        table { width: 100%; border-collapse: collapse; font-size: 13px; }
        th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid #334155; }
        th { color: #38bdf8; }
        tr.WARNING td:nth-child(3) { color: var(--warn); }
        tr.CRITICAL td:nth-child(3) { color: var(--crit); font-weight: bold; }
        a { color: #38bdf8; }
    </style>
</head>
<body>
    <h1>🛡️ Smart Sentinel</h1>
    <div id="status" class="NORMAL">NORMAL<small>Sin anomalías</small></div>
    <div class="container">
        <div class="video-box"><img src="/video_feed" alt="Cámara en directo"/></div>
        <div class="events-box">
            <h2 style="margin-top:0">Historial de eventos</h2>
            <table id="events-table">
                <thead><tr><th>Hora</th><th>Tipo</th><th>Nivel</th><th>Descripción</th><th></th></tr></thead>
                <tbody></tbody>
            </table>
        </div>
    </div>
    <script>
        function cell(tr, value) {
            const td = document.createElement('td');
            td.textContent = value;   // textContent: evita inyección HTML
            tr.appendChild(td);
            return td;
        }
        async function refresh() {
            try {
                const st = await (await fetch('/api/status')).json();
                const box = document.getElementById('status');
                box.className = st.level;
                box.replaceChildren(document.createTextNode(st.level + (st.acknowledged ? ' (reconocida)' : '')));
                const small = document.createElement('small');
                small.textContent = st.summary + ' · personas: ' + st.persons;
                box.appendChild(small);

                const events = await (await fetch('/api/events')).json();
                document.querySelector('#events-table tbody').replaceChildren(...events.map(e => {
                    const tr = document.createElement('tr');
                    tr.className = e.severity;
                    cell(tr, e.timestamp.slice(11));
                    cell(tr, e.event_type.replace('ANOMALIA_', ''));
                    cell(tr, e.severity);
                    cell(tr, e.description);
                    const td = document.createElement('td');
                    if (e.image) {
                        const a = document.createElement('a');
                        a.href = '/captures/' + encodeURIComponent(e.image);
                        a.target = '_blank';
                        a.textContent = '📷';
                        td.appendChild(a);
                    }
                    tr.appendChild(td);
                    return tr;
                }));
            } catch (err) {}
        }
        setInterval(refresh, 1500);
        refresh();
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


def generate_feed():
    while True:
        frame = frame_buffer
        if frame is not None:
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.05)  # ~20 fps; sin esto el bucle consume una CPU entera


@app.route('/video_feed')
def video_feed():
    return Response(generate_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/status')
def get_status():
    return jsonify(status)


@app.route('/api/events')
def get_events():
    try:
        with sqlite3.connect(Config.DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT id, timestamp, event_type, description, image_path, severity "
                "FROM events ORDER BY id DESC LIMIT 30").fetchall()
        return jsonify([{**{k: r[k] for k in ("id", "timestamp", "event_type", "description", "severity")},
                         "image": os.path.basename(r["image_path"] or "")} for r in rows])
    except Exception:
        return jsonify([])


@app.route('/captures/<path:filename>')
def captures(filename):
    # send_from_directory rechaza rutas que salgan de la carpeta (../)
    return send_from_directory(Config.CAPTURES_DIR, filename)


def run_dashboard():
    app.run(host=Config.DASHBOARD_HOST, port=Config.DASHBOARD_PORT, debug=False, use_reloader=False)
