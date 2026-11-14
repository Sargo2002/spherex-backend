"""SPHEREx SkyFlow backend: real SPHEREx data from NASA/IPAC IRSA (QR3 deep spectral images)."""
import base64, io, os, sqlite3, time
from urllib.parse import urlparse

# ============================================================
# Render-এ Proxy বাইপাস করার জন্য (502 Error সমাধান)
# ============================================================
os.environ['NO_PROXY'] = 'irsa.ipac.caltech.edu'
os.environ['no_proxy'] = 'irsa.ipac.caltech.edu'

import numpy as np
import requests
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.io.votable import parse_single_table
from astropy.time import Time
from astropy.visualization import AsinhStretch, ZScaleInterval
from astropy.wcs import WCS
from astroquery.ipac.irsa import Irsa
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
from scipy.ndimage import map_coordinates

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

DB = os.environ.get("DB_PATH", "spherex_live.db")

# 🔴 পরিবর্তন ১: QR2 থেকে QR3 Deep এ আপডেট করা হলো (এখানে ছবি বেশি পাওয়া যাবে)
COLLECTION = "spherex_qr3_deep" 

OUT_PX = 220
_cache = {}


def db():
    conn = sqlite3.connect(DB)
    conn.execute("""CREATE TABLE IF NOT EXISTS anomaly_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT, target_name TEXT, ra REAL, dec REAL,
        shift_arcsec REAL, notes TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    return conn


def err(msg, code=400):
    return jsonify({"error": str(msg)}), code


@app.route("/")
def index():
    return jsonify({
        "status": "SPHEREx SkyFlow API is live",
        "source": "NASA/IPAC IRSA — spherex_qr3_deep",
        "endpoints": [
            "/api/resolve?name=... or ?ra=...&dec=...",
            "/api/observations?ra=...&dec=...&fov=3&array=all&limit=12",
            "/api/cutout?url=...&ra=...&dec=...&fov=3",
            "/api/log-anomaly (POST)",
            "/api/get-logs"
        ]
    })


@app.route("/api/resolve")
def resolve():
    try:
        if request.args.get("name"):
            c = SkyCoord.from_name(request.args["name"])
        else:
            ra, dec = request.args["ra"].strip(), request.args["dec"].strip()
            try:
                c = SkyCoord(float(ra) * u.deg, float(dec) * u.deg)
            except ValueError:
                c = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
        return jsonify({"ra": c.ra.deg, "dec": c.dec.deg})
    except Exception as e:
        return err(f"Could not resolve target: {e}")


@app.route("/api/observations")
def observations():
    try:
        ra, dec = float(request.args["ra"]), float(request.args["dec"])
        fov = float(request.args.get("fov", 3))
        array = request.args.get("array", "all")
        
        # 🔴 পরিবর্তন ২: লিমিট ৪০ থেকে বাড়িয়ে ১০০ করা হলো (বেশি ছবি পাওয়ার জন্য)
        limit = min(int(request.args.get("limit", 12)), 100) 
        
        key = (round(ra, 4), round(dec, 4), round(fov, 1))
        if key in _cache and time.time() - _cache[key][0] < 900:
            rows = _cache[key][1]
        else:
            coord = SkyCoord(ra * u.deg, dec * u.deg)
            radius = max(fov / 120, 0.01)
            try:
                tab = Irsa.query_sia(pos=(coord, radius * u.deg), collection=COLLECTION)
            except Exception:
                resp = requests.get("https://irsa.ipac.caltech.edu/SIA", timeout=60, params={
                    "COLLECTION": COLLECTION, "POS": f"circle {ra} {dec} {radius}", "RESPONSEFORMAT": "votable"},
                    proxies={"http": None, "https": None})
                resp.raise_for_status()
                tab = parse_single_table(io.BytesIO(resp.content)).to_table()
            rows = []
            for r in tab:
                url = str(r["access_url"])
                parts = url.split("?")[0].split("/")
                det = parts[-2] if parts[-2].isdigit() else None
                rows.append({"url": url, "array": det, "mjd": float(r["t_min"]),
                             "obs_id": str(r["obs_id"]) if "obs_id" in tab.colnames else parts[-1]})
            rows.sort(key=lambda x: x["mjd"])
            _cache[key] = (time.time(), rows)
        if array != "all":
            rows = [r for r in rows if r["array"] == array]
        total = len(rows)
        if total > limit:
            rows = [rows[int(i * (total - 1) / (limit - 1))] for i in range(limit)] if limit > 1 else rows[:1]
        for r in rows:
            r["utc"] = Time(r["mjd"], format="mjd").iso[:16]
        return jsonify({"collection": COLLECTION, "total": total, "observations": rows})
    except Exception as e:
        return err(f"IRSA query failed: {e}", 502)


@app.route("/api/cutout")
def cutout():
    try:
        url = request.args["url"]
        if urlparse(url).hostname != "irsa.ipac.caltech.edu":
            return err("Only IRSA URLs are allowed")
        ra, dec = float(request.args["ra"]), float(request.args["dec"])
        fov = float(request.args["fov"])
        size_deg = fov / 60 * 1.6
        r = requests.get(url.split("?")[0], params={"center": f"{ra},{dec}", "size": f"{size_deg}"}, timeout=90, 
                         proxies={"http": None, "https": None})
        r.raise_for_status()
        with fits.open(io.BytesIO(r.content)) as h:
            img = h["IMAGE"].data.astype(float)
            hdr = h["IMAGE"].header
            flags = h["FLAGS"].data if "FLAGS" in h else None
            var = h["VARIANCE"].data if "VARIANCE" in h else None
            wcs = WCS(hdr, naxis=2)
            wave = None
            try:
                x0, y0 = wcs.world_to_pixel(SkyCoord(ra * u.deg, dec * u.deg))
                ww = WCS(hdr, fobj=h, key="W")
                w, _ = ww.pixel_to_world(x0, y0)
                wave = float(w.to(u.um).value)
            except Exception:
                pass
        scale = fov * 60 / OUT_PX
        out = WCS(naxis=2)
        out.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        out.wcs.crval = [ra, dec]
        out.wcs.crpix = [OUT_PX / 2 + .5, OUT_PX / 2 + .5]
        out.wcs.cdelt = [-scale / 3600, scale / 3600]
        yy, xx = np.mgrid[0:OUT_PX, 0:OUT_PX]
        sky = out.pixel_to_world(xx, yy)
        px, py = wcs.world_to_pixel(sky)
        grid = map_coordinates(img, [py, px], order=1, cval=np.nan)
        cx, cy = wcs.world_to_pixel(SkyCoord(ra * u.deg, dec * u.deg))
        ci, cj = int(round(float(cy))), int(round(float(cx)))
        sl = (slice(max(ci - 1, 0), ci + 2), slice(max(cj - 1, 0), cj + 2))
        flux = float(np.nanmedian(img[sl]))
        ferr = float(np.sqrt(np.nanmedian(var[sl]))) if var is not None else None
        flagged = float(np.mean(flags != 0)) if flags is not None else None
        good = np.isfinite(grid)
        if not good.any():
            return err("Target not inside this image", 404)
        lo, hi = ZScaleInterval().get_limits(grid[good])
        norm = np.clip((grid - lo) / max(hi - lo, 1e-12), 0, 1)
        norm = np.nan_to_num(AsinhStretch(a=0.3)(norm))
        buf = io.BytesIO()
        Image.fromarray((norm[::-1] * 255).astype("uint8")).save(buf, "PNG")
        clean = lambda v: v if v is None or np.isfinite(v) else None
        return jsonify({"png": base64.b64encode(buf.getvalue()).decode(), "wavelength_um": wave,
                        "flux_mjysr": clean(flux), "flux_err": clean(ferr), "flagged_fraction": clean(flagged)})
    except Exception as e:
        return err(f"Cutout failed: {e}", 502)


@app.route("/api/log-anomaly", methods=["POST"])
def log_anomaly():
    d = request.get_json(force=True)
    with db() as c:
        c.execute("INSERT INTO anomaly_logs (target_name, ra, dec, shift_arcsec, notes) VALUES (?,?,?,?,?)",
                  (d.get("target"), d.get("ra"), d.get("dec"), d.get("shift"), d.get("notes")))
    return jsonify({"status": "saved"})


@app.route("/api/get-logs")
def get_logs():
    with db() as c:
        rows = c.execute("SELECT id,target_name,ra,dec,shift_arcsec,notes,created_at FROM anomaly_logs ORDER BY id DESC LIMIT 50").fetchall()
    return jsonify([dict(zip(["id", "target", "ra", "dec", "shift", "notes", "time"], r)) for r in rows])


if __name__ == "__main__":
    db().close()
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=False, host="0.0.0.0", port=port)
