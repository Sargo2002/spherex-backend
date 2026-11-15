"""SPHEREx SkyFlow backend: real SPHEREx data from NASA/IPAC IRSA."""
import base64, io, os, sqlite3, time
from urllib.parse import urlparse

# Render Proxy বন্ধ করার জন্য
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
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_compress import Compress
from PIL import Image
from scipy.ndimage import map_coordinates

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
Compress(app)  # ডেটা কম্প্রেস করে দ্রুত পাঠানোর জন্য

DB = os.environ.get("DB_PATH", "spherex_live.db")
COLLECTION = "spherex_qr3"  
OUT_PX = 220                        
_cache = {}                         

# গ্লোবাল এরর হ্যান্ডলার (সার্ভার ক্র্যাশ রোধ করতে)
@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

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
    return jsonify({"status": "SPHEREx SkyFlow API is live", "source": "NASA/IPAC IRSA"})

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
        limit = min(int(request.args.get("limit", 12)), 40)
        
        radius = max(fov / 60.0, 0.3) 
        
        params = {
            "COLLECTION": COLLECTION,
            "POS": f"circle {ra} {dec} {radius}",
            "RESPONSEFORMAT": "votable"
        }
        
        # রিট্রাই লজিক এবং বড় টাইমআউট
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(max_retries=3)
        session.mount('https://', adapter)
        
        resp = session.get("https://irsa.ipac.caltech.edu/SIA", params=params, timeout=180)
        resp.raise_for_status()
        
        tab = parse_single_table(io.BytesIO(resp.content)).to_table()
        
        rows = []
        for r in tab:
            url = str(r["access_url"])
            parts = url.split("?")[0].split("/")
            det = parts[-2] if parts[-2].isdigit() else "unknown"
            rows.append({
                "url": url, 
                "array": det, 
                "mjd": float(r["t_min"]),
                "obs_id": str(r["obs_id"]) if "obs_id" in tab.colnames else parts[-1]
            })
        
        rows.sort(key=lambda x: x["mjd"])
        
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
        ra, dec = float(request.args["ra"]), float(request.args["dec"])
        fov = float(request.args["fov"])
        size_deg = fov / 60 * 1.5 
        
        r = requests.get(url.split("?")[0], params={"center": f"{ra},{dec}", "size": f"{size_deg}"}, timeout=300)
        r.raise_for_status()
        
        with fits.open(io.BytesIO(r.content)) as h:
            img = h["IMAGE"].data.astype(float)
            hdr = h["IMAGE"].header
            wcs = WCS(hdr, naxis=2)
            wave = None
                
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
        
        good = np.isfinite(grid)
        if not good.any():
            return err("Target not inside this image", 404)
            
        lo, hi = ZScaleInterval().get_limits(grid[good])
        norm = np.clip((grid - lo) / max(hi - lo, 1e-12), 0, 1)
        norm = np.nan_to_num(AsinhStretch(a=0.3)(norm))
        
        buf = io.BytesIO()
        Image.fromarray((norm[::-1] * 255).astype("uint8")).save(buf, "PNG")
        
        return jsonify({
            "png": base64.b64encode(buf.getvalue()).decode(), 
            "wavelength_um": wave,
            "flux_mjysr": float(np.nanmedian(img)), 
            "flux_err": None, 
            "flagged_fraction": None
        })
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
