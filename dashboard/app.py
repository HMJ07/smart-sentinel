import hmac
import os
import secrets
import sqlite3
import time
from datetime import timedelta
from pathlib import Path

import cv2
from flask import (Flask, Response, jsonify, redirect, render_template_string, request,
                   send_from_directory, session, url_for)

from config.settings import Config

app = Flask(__name__)
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    PERMANENT_SESSION_LIFETIME=timedelta(hours=12),
    SENTINEL_PASSWORD=None,
)
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
    <form method="post" action="/logout" style="float:right"><button style="background:#1e293b;color:#94a3b8;border:1px solid #334155;border-radius:6px;padding:6px 12px;cursor:pointer">Cerrar sesión</button></form>
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



# ------------------------------------------------------------------ autenticación
MAX_FAILED_LOGINS = 5
LOCKOUT_SECONDS = 60
_failed = {}   # ip -> (fallos, instante del primer fallo)

LOGIN_TEMPLATE = """
<!DOCTYPE html>
<html lang="es"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Smart Sentinel - Acceso</title>
<style>
 body { font-family: ui-monospace, monospace; background:#0f172a; color:#e2e8f0; display:grid; place-items:center; height:100vh; margin:0; }
 form { background:#1e293b; padding:28px; border-radius:10px; width:300px; border-left:6px solid #38bdf8; }
 h1 { color:#38bdf8; font-size:20px; margin:0 0 16px; }
 input, button { width:100%; box-sizing:border-box; padding:10px; margin-top:10px; border-radius:6px; border:1px solid #334155; font:inherit; }
 input { background:#0f172a; color:#e2e8f0; } button { background:#38bdf8; color:#0f172a; font-weight:bold; cursor:pointer; }
 .err { color:#ef4444; margin-top:10px; font-size:13px; }
</style></head><body>
<form method="post" action="/login">
 <form method="post" action="/logout" style="float:right"><button style="background:#1e293b;color:#94a3b8;border:1px solid #334155;border-radius:6px;padding:6px 12px;cursor:pointer">Cerrar sesión</button></form>
    <h1>🛡️ Smart Sentinel</h1>
 <label for="p">Clave de acceso</label>
 <input id="p" type="password" name="password" autofocus autocomplete="current-password" required>
 <button type="submit">Entrar</button>
 {% if error %}<div class="err">{{ error }}</div>{% endif %}
</form></body></html>
"""


def configure_auth():
    """Fija la clave del dashboard. Devuelve (clave, es_automatica, ruta_del_archivo).

    Prioridad: SENTINEL_DASHBOARD_PASSWORD > archivo guardado > clave aleatoria nueva (se guarda).
    """
    data_dir = Path(Config.DATA_DIR)
    key_file = data_dir / "dashboard_password.txt"

    if Config.DASHBOARD_PASSWORD:
        password, automatic = Config.DASHBOARD_PASSWORD, False
    elif key_file.exists() and key_file.read_text(encoding="utf-8").strip():
        password, automatic = key_file.read_text(encoding="utf-8").strip(), True
    else:
        password, automatic = secrets.token_urlsafe(9), True
        key_file.write_text(password, encoding="utf-8")
        try:
            key_file.chmod(0o600)
        except OSError:
            pass

    secret_file = data_dir / "session_secret"
    if not secret_file.exists():
        secret_file.write_text(secrets.token_hex(32), encoding="utf-8")
        try:
            secret_file.chmod(0o600)
        except OSError:
            pass
    app.secret_key = secret_file.read_text(encoding="utf-8").strip()
    app.config["SENTINEL_PASSWORD"] = password
    return password, automatic, str(key_file)


def _locked_out(ip):
    fails, first = _failed.get(ip, (0, 0.0))
    if fails >= MAX_FAILED_LOGINS:
        if time.time() - first < LOCKOUT_SECONDS:
            return True
        _failed.pop(ip, None)
    return False


def _register_failure(ip):
    fails, first = _failed.get(ip, (0, time.time()))
    _failed[ip] = (fails + 1, first)


@app.before_request
def require_login():
    if request.endpoint in ("login", "static"):
        return None
    if app.config["SENTINEL_PASSWORD"] is None:      # nunca servir nada si la clave no está configurada
        return Response("Autenticación no configurada", status=503)
    if session.get("ok"):
        return None
    if request.path.startswith(("/api/", "/video_feed", "/captures/")):
        return jsonify({"error": "no autenticado"}), 401
    return redirect(url_for("login"))


@app.after_request
def security_headers(resp):
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["Cache-Control"] = "no-store"
    resp.headers["Content-Security-Policy"] = (
        "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
        "script-src 'self' 'unsafe-inline'; frame-ancestors 'none'")
    return resp


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template_string(LOGIN_TEMPLATE, error=None)
    ip = request.remote_addr or "?"
    if _locked_out(ip):
        return render_template_string(LOGIN_TEMPLATE, error="Demasiados intentos. Espera un minuto."), 429
    supplied = request.form.get("password", "").encode("utf-8")
    expected = app.config["SENTINEL_PASSWORD"].encode("utf-8")
    if hmac.compare_digest(supplied, expected):
        _failed.pop(ip, None)
        session.clear()
        session["ok"] = True
        session.permanent = True
        return redirect(url_for("index"))
    _register_failure(ip)
    return render_template_string(LOGIN_TEMPLATE, error="Clave incorrecta."), 401


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for("login"))


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
    if app.config["SENTINEL_PASSWORD"] is None:
        configure_auth()
    app.run(host=Config.DASHBOARD_HOST, port=Config.DASHBOARD_PORT, debug=False, use_reloader=False)
