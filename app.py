from flask import Flask, render_template, request
from db import get_conn, init_db, reset_all_data, set_live_feed_enabled
import config as cfg

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/latest")
def api_latest():
    with get_conn() as conn:
        status = conn.execute(
            "SELECT * FROM current_status WHERE id = 1"
        ).fetchone()
        latest_event = conn.execute(
            "SELECT * FROM events ORDER BY id DESC LIMIT 1"
        ).fetchone()

    return {
        "status": dict(status) if status else {},
        "latest_event": dict(latest_event) if latest_event else None,
    }


@app.route("/api/events")
def api_events():
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM events ORDER BY id DESC LIMIT 25"
        ).fetchall()
    return [dict(row) for row in rows]


@app.route("/api/live-feed-toggle", methods=["POST"])
def api_live_feed_toggle():
    payload = request.get_json(silent=True) or {}
    enabled = bool(payload.get("enabled", False))
    set_live_feed_enabled(enabled)
    return {"ok": True, "enabled": enabled}


@app.route("/api/reset", methods=["POST"])
def api_reset():
    reset_all_data(delete_images=True)
    return {"ok": True}


if __name__ == "__main__":
    init_db()
    app.run(host=cfg.HOST, port=cfg.PORT, debug=True)
