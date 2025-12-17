# app_1.py
# Digital Twin Telescope — Unified Dashboard (Edition 7.3)
# Robust: creates safe fallback PSF/sensor if missing, avoids Dash id-not-found errors

import os
import io
import json
import webbrowser
import traceback
from math import isfinite
from datetime import datetime, timezone

import numpy as np

# optional numerical/image libs (graceful fallback)
try:
    from scipy.signal import fftconvolve
    from scipy.ndimage import gaussian_filter
except Exception:
    fftconvolve = None
    def gaussian_filter(x, sigma=1.0):
        # simple gaussian fallback using numpy (very small kernel)
        if sigma <= 0:
            return x
        # tiny separable gaussian blur approximate
        from math import exp, sqrt, pi
        ksize = max(3, int(6 * sigma))
        if ksize % 2 == 0:
            ksize += 1
        ax = np.arange(-ksize//2 + 1., ksize//2 + 1.)
        kernel = np.exp(-(ax**2)/(2*sigma*sigma))
        kernel = kernel / np.sum(kernel)
        # separable conv
        tmp = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis=0, arr=x)
        res = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis=1, arr=tmp)
        return res

# astropy for fits display and writes (optional)
try:
    from astropy.io import fits
    from astropy.visualization import simple_norm
except Exception:
    fits = None
    def simple_norm(x, **kwargs):
        return lambda arr: arr

import dash
from dash import dcc, html, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go

# ----------------------------
# Metadata & example paths
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_TITLE = "Digital Twin Telescope — Dashboard"
EXAMPLE_FILES = {
    "psf": os.path.join(BASE_DIR, "final_combined_psf.fits"),
    "sensor": os.path.join(BASE_DIR, "sensor_output.fits"),
    "catalog": os.path.join(BASE_DIR, "star_catalog.json"),
    "merged_catalog": os.path.join(BASE_DIR, "merged_sky_catalog.json"),
    "planet_snapshot": os.path.join(BASE_DIR, "planet_catalog.json"),
}
OBSERVER_NAME = "Chennai, Tamil Nadu"

# ----------------------------
# Fallback asset creation (safe, non-destructive)
# ----------------------------
def create_fallback_psf(path, size=65):
    """Create a synthetic Gaussian PSF and write to disk (if astropy available writes FITS)."""
    x = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, x)
    sigma = 0.08
    psf = np.exp(-(xx**2 + yy**2)/(2*sigma**2))
    psf /= psf.sum()
    try:
        if fits is not None:
            hdu = fits.PrimaryHDU(psf.astype(np.float32))
            hdu.writeto(path, overwrite=True)
        else:
            np.save(path + ".npy", psf.astype(np.float32))
    except Exception:
        # best-effort
        np.save(path + ".npy", psf.astype(np.float32))
    return psf

def create_fallback_sensor(path, shape=(256,256)):
    """Create a synthetic sensor image (stars + noise)."""
    img = np.zeros(shape, dtype=float)
    rng = np.random.default_rng(0)
    # place several bright points
    for _ in range(180):
        r = rng.integers(0, shape[0])
        c = rng.integers(0, shape[1])
        intensity = rng.uniform(4000, 8000)
        img[r % shape[0], c % shape[1]] += intensity
    # slight gaussian blur to make PSF-like blobs
    img = gaussian_filter(img, sigma=1.5)
    # add noise + background
    img += rng.poisson(15, size=shape)
    img += rng.normal(0, 3, size=shape)
    try:
        if fits is not None:
            hdu = fits.PrimaryHDU(img.astype(np.float32))
            hdu.writeto(path, overwrite=True)
        else:
            np.save(path + ".npy", img.astype(np.float32))
    except Exception:
        np.save(path + ".npy", img.astype(np.float32))
    return img

def ensure_fallback_assets():
    # create missing PSF
    if not os.path.exists(EXAMPLE_FILES['psf']):
        print("PSF file missing; creating fallback PSF...")
        create_fallback_psf(EXAMPLE_FILES['psf'])
    # create missing sensor
    if not os.path.exists(EXAMPLE_FILES['sensor']):
        print("Sensor file missing; creating fallback sensor image...")
        create_fallback_sensor(EXAMPLE_FILES['sensor'])

# Run fallback creation early to avoid missing-file issues
ensure_fallback_assets()

# ----------------------------
# Utilities
# ----------------------------
def safe_load_fits(path):
    if not os.path.exists(path) or fits is None:
        # try .npy fallback
        if os.path.exists(path + ".npy"):
            try:
                return np.load(path + ".npy")
            except Exception:
                return None
        return None
    try:
        data = fits.getdata(path)
        return np.array(data, dtype=float)
    except Exception:
        return None

def ensure_2d_array(arr):
    """Return a 2D numpy array if possible, else None."""
    try:
        a = np.array(arr, dtype=float)
    except Exception:
        return None
    if a.size == 0:
        return None
    if a.ndim == 2:
        return a
    if a.ndim > 2:
        return a[0]
    if a.ndim == 1:
        n = a.size
        root = int(np.sqrt(n))
        if root * root == n:
            return a.reshape((root, root))
        return a.reshape((1, n))
    return None

def normalize_for_plot(img, stretch=99.5):
    try:
        img = np.array(img, dtype=float)
    except Exception:
        return np.zeros((2,2))
    if img.ndim == 0:
        img = np.array([[float(img)]])
    if img.ndim == 1:
        img = img.reshape((1, img.size))
    p = np.percentile(img, [0.1, stretch])
    if p[1] == p[0]:
        return np.clip(img, 0, None)
    try:
        norm = simple_norm(img, stretch='linear', min_cut=p[0], max_cut=p[1])
        return norm(img)
    except Exception:
        return np.clip(img, 0, None)

def compute_mtf(psf):
    try:
        psf2d = ensure_2d_array(psf)
        if psf2d is None:
            return np.zeros((2,2))
        otf = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(psf2d)))
        mtf = np.abs(otf)
        if mtf.max() != 0:
            mtf /= mtf.max()
        return mtf
    except Exception:
        return np.zeros((2,2))

def convolve_with_psf(image, psf):
    try:
        psf2d = ensure_2d_array(psf)
        if psf2d is None or fftconvolve is None:
            # fallback: gaussian blur approximate
            return gaussian_filter(image, sigma=1.0)
        pnorm = psf2d / (psf2d.sum() + 1e-16)
        return fftconvolve(image, pnorm, mode='same')
    except Exception:
        return image

def add_sensor_noise(image, sky_background=15, gain=1.4, read_noise=5, dark_current_rate=0.01, exposure_time=2.0):
    try:
        signal = np.clip(image, 0, None)
        total = signal + sky_background
        shot = np.random.poisson(np.clip(total, 0, None)).astype(float)
        dark = np.random.poisson(dark_current_rate * exposure_time, size=signal.shape)
        read = np.random.normal(0, read_noise, size=signal.shape)
        return (shot + dark + read) / (gain + 1e-12)
    except Exception:
        return image + np.random.normal(0, 1.0, size=image.shape)

def fits_bytes(arr):
    if fits is None:
        return b""
    hdu = fits.PrimaryHDU(np.array(arr, dtype=np.float32))
    mem = io.BytesIO()
    hdu.writeto(mem)
    mem.seek(0)
    return mem.read()

# ----------------------------
# Preload assets (if present)
# ----------------------------
PSF0 = safe_load_fits(EXAMPLE_FILES['psf'])
SENSOR0 = safe_load_fits(EXAMPLE_FILES['sensor'])

CATALOG0 = None
if os.path.exists(EXAMPLE_FILES['merged_catalog']):
    try:
        with open(EXAMPLE_FILES['merged_catalog']) as f:
            CATALOG0 = json.load(f)
    except Exception:
        CATALOG0 = None
elif os.path.exists(EXAMPLE_FILES['catalog']):
    try:
        with open(EXAMPLE_FILES['catalog']) as f:
            CATALOG0 = json.load(f)
    except Exception:
        CATALOG0 = None

PLANETS_SNAPSHOT = None
if os.path.exists(EXAMPLE_FILES['planet_snapshot']):
    try:
        with open(EXAMPLE_FILES['planet_snapshot']) as f:
            PLANETS_SNAPSHOT = json.load(f)
    except Exception:
        PLANETS_SNAPSHOT = None

# ----------------------------
# Dash app (Bootstrap dark theme + optional assets/dashboard.css)
# ----------------------------
external_stylesheets = [dbc.themes.DARKLY]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

server = app.server
app.title = APP_TITLE
px.defaults.template = "plotly_dark"

# ----------------------------
# Smart label placement helpers
# ----------------------------
def smart_text_position(ra, dec):
    if dec > 45:
        return "top center"
    if dec > 10:
        return "top center"
    if dec < -45:
        return "bottom center"
    if dec < -10:
        return "bottom center"
    return "middle right"

def apply_simple_collision_avoid(pts, dec_range):
    if not pts:
        return []
    arr = sorted(pts, key=lambda p: p["dec"])
    min_sep = max(0.4, (dec_range[1] - dec_range[0]) / 60.0)
    for i in range(1, len(arr)):
        prev = arr[i-1]; cur = arr[i]
        required = prev["dec"] + min_sep
        if cur["dec"] < required:
            cur["dec"] = required
    for p in arr:
        if p["dec"] < dec_range[0]:
            p["dec"] = dec_range[0]
        if p["dec"] > dec_range[1]:
            p["dec"] = dec_range[1]
    return arr

# ----------------------------
# Layout UI components
# ----------------------------
controls = dbc.Card(
    [
        dbc.CardHeader(html.H5("Sky Controls")),
        dbc.CardBody([
            html.Div(f"Observer: {OBSERVER_NAME}", className="observer-line"),
            html.Hr(),
            dbc.Label("RA Range (deg)"),
            dcc.RangeSlider(0, 360, 10, value=[0,360], id="ra-slider"),
            html.Br(),
            dbc.Label("DEC Range (deg)"),
            dcc.RangeSlider(-90, 90, 5, value=[-90,90], id="dec-slider"),
            html.Br(),
            dbc.Label("Max Magnitude (lower = brighter)"),
            dcc.Slider(0, 20, 0.5, value=14, id="mag-slider"),
            html.Br(),
            dcc.Checklist(options=[{"label":" Show sample planets","value":"YES"}], value=["YES"], id="planet-toggle"),
            html.Br(),
            dbc.Button("Reset View", id="reset-btn", color="secondary", size="sm")
        ])
    ], color="dark"
)

sim_controls = dbc.Card(
    [
        dbc.CardHeader(html.H5("Optical Simulation Controls")),
        dbc.CardBody([
            dbc.Label("Choose example file"),
            dcc.Dropdown(id='load_choice',
                         options=[{'label':k,'value':k} for k in ["psf","sensor","catalog","planet_snapshot"]],
                         value='psf'),
            html.Div(id='load_status', style={'marginTop':'8px'}),
            html.Hr(),
            dbc.Label("Exposure time (s)"),
            dcc.Slider(id='exposure_time', min=0.1, max=10, step=0.1, value=2),
            dbc.Label("Seeing FWHM (arcsec)", className='mt-2'),
            dcc.Slider(id='seeing_fwhm', min=0.3, max=4.0, step=0.1, value=1.8),
            dbc.Label("Telescope drift (px/frame)", className='mt-2'),
            dcc.Slider(id='drift_pix', min=0.0, max=6.0, step=0.1, value=0.5),
            dbc.Label("Sky background (e/pix)", className='mt-2'),
            dbc.Input(id='sky_bg', type='number', value=15),
            html.Br(),
            dbc.Button("Simulate single exposure", id='btn_single', color='primary', className='me-2'),
            dbc.Button("Simulate multiframe (6 frames)", id='btn_multi', color='secondary'),
            html.Div(id='simulate_msg', style={'marginTop':'8px'})
        ])
    ], color='dark'
)

left = dbc.Col([controls, html.Br(), sim_controls], width=3)

right = dbc.Col([
    html.H2(APP_TITLE, className="app-title"),
    dcc.Tabs(id="main-tabs", value="tab-sky", children=[
        dcc.Tab(label="Live Sky Viewer", value="tab-sky"),
        dcc.Tab(label="Optical Simulation", value="tab-sim"),
    ]),
    html.Div(id="tab-content", className="tab-content")
], width=9)

# Top-level layout: include cosmetic divs and hidden placeholders for PSF/Sensor graphs so IDs exist
app.layout = html.Div(
    className="milkyway-on satellite-on jwst-theme parallax-on",
    children=[
        html.Div(className="starfield"),
        html.Div(className="milkyway"),
        html.Div(className="satellite"),
        html.Div(className="jwst-hex"),

        # Hidden placeholders so Dash always knows these IDs exist (prevents "ID not found" errors)
        html.Div(style={'display':'none'}, children=[
            dcc.Graph(id='psf_graph'),
            dcc.Graph(id='mtf_graph'),
            dcc.Graph(id='sensor_graph'),
        ]),

        dbc.Container([
            dbc.Row([left, right], align="start"),
            dcc.Store(id='store_psf', data=PSF0.tolist() if PSF0 is not None else None),
            dcc.Store(id='store_sensor', data=SENSOR0.tolist() if SENSOR0 is not None else None),
            dcc.Store(id='store_sim'),
            dcc.Download(id='download_fits')
        ], fluid=True, style={'paddingTop':'8px', 'paddingBottom':'40px', 'backgroundColor':'#05060a'})
    ]
)

# ----------------------------
# Graph builders (unchanged logic)
# ----------------------------
def build_optical_layout():
    return html.Div([
        dbc.Row([
            dbc.Col(dbc.Card([dbc.CardHeader("PSF (preview)"), dbc.CardBody(dcc.Graph(id='psf_graph'))]), width=6),
            dbc.Col(dbc.Card([dbc.CardHeader("MTF (2D)"), dbc.CardBody(dcc.Graph(id='mtf_graph'))]), width=6)
        ], className="mb-3"),
        dbc.Row([
            dbc.Col(dbc.Card([dbc.CardHeader("Sensor / Simulated Image"),
                              dbc.CardBody(dcc.Graph(id='sensor_graph', style={'height':'520px'})),
                              dbc.CardFooter(html.Div([dbc.Button("Download FITS", id='dl_image', color='link'),
                                                       html.Span(id='sim_info', style={'float':'right'})]))]), width=12)
        ]),
        html.Hr()
    ])

def make_2d_sky_figure(catalog, planet_snapshot, ra_range, dec_range, mag_limit, show_planets):
    ra_list = []; dec_list = []; mag_list = []
    for c in catalog:
        try:
            ra_val = float(c.get("ra_deg", c.get("ra", np.nan)))
            dec_val = float(c.get("dec_deg", c.get("dec", np.nan)))
            mag_val = float(c.get("brightness", c.get("mag", 15.0)))
        except Exception:
            continue
        if not (isfinite(ra_val) and isfinite(dec_val) and isfinite(mag_val)):
            continue
        ra_list.append(ra_val); dec_list.append(dec_val); mag_list.append(mag_val)

    if len(ra_list) == 0:
        return go.Figure().update_layout(title="No stars in catalog")

    ra = np.array(ra_list, dtype=float); dec = np.array(dec_list, dtype=float); mag = np.array(mag_list, dtype=float)
    mask = (ra >= ra_range[0]) & (ra <= ra_range[1]) & (dec >= dec_range[0]) & (dec <= dec_range[1]) & (mag <= mag_limit)
    ra = ra[mask]; dec = dec[mask]; mag = mag[mask]

    fig = px.scatter(x=ra, y=dec, color=mag, color_continuous_scale='viridis',
                     labels={"x":"RA (deg)", "y":"DEC (deg)"},
                     title=f"2D Sky Map — {len(ra)} stars displayed")
    fig.update_layout(template='plotly_dark', height=650, margin=dict(l=40, r=80, t=50, b=40))
    fig.update_yaxes(autorange='reversed', scaleanchor="x", scaleratio=1)

    if show_planets and planet_snapshot:
        pts = []
        for p in planet_snapshot:
            if "ra_deg" in p and "dec_deg" in p:
                try:
                    pr = float(p["ra_deg"]); pd = float(p["dec_deg"])
                except Exception:
                    continue
                if (pr >= ra_range[0] and pr <= ra_range[1] and pd >= dec_range[0] and pd <= dec_range[1]):
                    pts.append({"name": p.get("name",""), "ra": pr, "dec": pd})
        if pts:
            adjusted = apply_simple_collision_avoid(pts, dec_range)
            fig.add_trace(go.Scatter(
                x=[p["ra"] for p in adjusted],
                y=[p["dec"] for p in adjusted],
                mode="markers",
                marker=dict(size=12, color='orange', symbol='circle'),
                hoverinfo='text',
                hovertext=[p["name"] for p in adjusted],
                name="Planets",
                showlegend=True
            ))
            annotations = []
            for p in adjusted:
                pos = smart_text_position(p["ra"], p["dec"])
                ay = -18 if pos.startswith("top") else (18 if pos.startswith("bottom") else 0)
                ax = 0 if "center" in pos or "right" in pos else -10
                annotations.append(dict(
                    x=p["ra"], y=p["dec"], xref="x", yref="y",
                    text=str(p["name"]),
                    showarrow=True,
                    arrowhead=2,
                    ax=ax, ay=ay,
                    font=dict(color="white", size=11),
                    bgcolor="rgba(0,0,0,0.4)",
                    borderwidth=0
                ))
            if annotations:
                fig.update_layout(annotations=annotations)
    if getattr(fig.layout, "coloraxis", None) is not None:
        fig.update_layout(coloraxis_colorbar=dict(x=0.95))
    return fig

def make_3d_globe_figure(catalog):
    ra_vals = []; dec_vals = []; mag_vals = []
    for c in catalog:
        try:
            ra_val = float(c.get("ra_deg", c.get("ra", np.nan)))
            dec_val = float(c.get("dec_deg", c.get("dec", np.nan)))
            mag_val = float(c.get("brightness", c.get("mag", 15.0)))
        except Exception:
            continue
        if not (isfinite(ra_val) and isfinite(dec_val) and isfinite(mag_val)):
            continue
        ra_vals.append(ra_val); dec_vals.append(dec_val); mag_vals.append(mag_val)

    if not ra_vals:
        return go.Figure().update_layout(title="No stars for 3D globe")

    ra = np.array(ra_vals); dec = np.array(dec_vals); mag = np.array(mag_vals)
    phi = np.radians(ra); theta = np.radians(90 - dec)
    x = np.sin(theta) * np.cos(phi); y = np.sin(theta) * np.sin(phi); z = np.cos(theta)
    brightness = 1.0 / (mag + 1e-6)
    bnorm = (brightness - brightness.min()) / (np.ptp(brightness) + 1e-12)
    size = 2 + 6*bnorm

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers',
                               marker=dict(size=size, color=bnorm, colorscale='Viridis', opacity=0.9)))
    fig.update_layout(template='plotly_dark', height=650,
                      scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)))
    return fig

# ----------------------------
# Callbacks (load / psf / simulate / render tab)
# ----------------------------
@app.callback(
    Output('load_status','children'),
    Output('store_psf','data'),
    Output('store_sensor','data'),
    Input('load_choice','value')
)
def load_example(choice):
    try:
        msgs = []
        psf_data = None
        sensor_data = None
        psf_path = EXAMPLE_FILES['psf']
        sensor_path = EXAMPLE_FILES['sensor']
        if os.path.exists(psf_path):
            arr = safe_load_fits(psf_path)
            if arr is not None:
                psf_data = arr.tolist(); msgs.append("PSF loaded ✅")
            else:
                msgs.append("PSF unreadable ❌")
        else:
            msgs.append(f"PSF missing: {os.path.basename(psf_path)}")
        if os.path.exists(sensor_path):
            arr = safe_load_fits(sensor_path)
            if arr is not None:
                sensor_data = arr.tolist(); msgs.append("Sensor loaded ✅")
            else:
                msgs.append("Sensor unreadable ❌")
        else:
            msgs.append(f"Sensor missing: {os.path.basename(sensor_path)}")
        return html.Ul([html.Li(m) for m in msgs]), psf_data, sensor_data
    except Exception:
        traceback.print_exc()
        return html.Ul([html.Li("Load error")]), None, None

@app.callback(
    Output('psf_graph','figure'),
    Output('mtf_graph','figure'),
    Input('store_psf','data')
)
def show_psf(psf_serial):
    try:
        if psf_serial is None:
            f = go.Figure(); f.update_layout(title='No PSF loaded', template='plotly_dark'); return f, f
        psf = ensure_2d_array(psf_serial)
        if psf is None:
            f = go.Figure(); f.update_layout(title='PSF parse error', template='plotly_dark'); return f, f
        fig_psf = px.imshow(normalize_for_plot(psf), origin='lower', color_continuous_scale='viridis')
        fig_psf.update_layout(title='PSF (normalized)', template='plotly_dark', height=350)
        mtf = compute_mtf(psf)
        fig_mtf = px.imshow(normalize_for_plot(mtf), origin='lower', color_continuous_scale='viridis')
        fig_mtf.update_layout(title='MTF', template='plotly_dark', height=350)
        return fig_psf, fig_mtf
    except Exception:
        traceback.print_exc()
        f = go.Figure(); f.update_layout(title='PSF error', template='plotly_dark'); return f, f

@app.callback(
    Output('sensor_graph','figure'),
    Output('store_sim','data'),
    Output('simulate_msg','children'),
    Input('btn_single','n_clicks'),
    Input('btn_multi','n_clicks'),
    State('store_psf','data'),
    State('store_sensor','data'),
    State('exposure_time','value'),
    State('seeing_fwhm','value'),
    State('drift_pix','value'),
    State('sky_bg','value')
)
def run_sim(n_single, n_multi, psf_serial, sensor_serial, exposure_time, seeing_fwhm, drift_pix, sky_bg):
    try:
        trigger = ctx.triggered_id
        psf = ensure_2d_array(psf_serial) if psf_serial is not None else None
        sensor = ensure_2d_array(sensor_serial) if sensor_serial is not None else None
        if sensor is not None:
            base = sensor.copy()
        else:
            base = np.zeros((256,256)); base[128 % base.shape[0], 128 % base.shape[1]] = 1e4
        if trigger == 'btn_single':
            if psf is not None:
                out = convolve_with_psf(base, psf)
            else:
                sigma = (seeing_fwhm / 0.4) / (2*np.sqrt(2*np.log(2))) if seeing_fwhm else 1.0
                out = gaussian_filter(base, sigma=sigma)
            noisy = add_sensor_noise(out, sky_background=sky_bg or 0, exposure_time=exposure_time or 1.0)
            fig = px.imshow(normalize_for_plot(noisy), origin='lower', color_continuous_scale='inferno')
            fig.update_layout(title='Simulated Single Exposure', template='plotly_dark', height=600)
            return fig, noisy.tolist(), "Single exposure simulated ✅"
        if trigger == 'btn_multi':
            frames = []
            for i in range(6):
                dy = int(np.round(np.random.uniform(- (drift_pix or 0), drift_pix or 0)))
                dx = int(np.round(np.random.uniform(- (drift_pix or 0), drift_pix or 0)))
                shifted = np.roll(np.roll(base, dy, axis=0), dx, axis=1)
                if psf is not None:
                    blurred = convolve_with_psf(shifted, psf)
                else:
                    sigma = (seeing_fwhm / 0.4) / (2*np.sqrt(2*np.log(2))) if seeing_fwhm else 1.0
                    blurred = gaussian_filter(shifted, sigma=sigma)
                noisy = add_sensor_noise(blurred, sky_background=sky_bg or 0, exposure_time=exposure_time or 1.0)
                frames.append(noisy)
            stacked = np.median(np.stack(frames), axis=0)
            fig = px.imshow(normalize_for_plot(stacked), origin='lower', color_continuous_scale='inferno')
            fig.update_layout(title=f'Stacked ({len(frames)}) exposures (median)', template='plotly_dark', height=600)
            return fig, stacked.tolist(), f'Multiframe simulated and stacked ({len(frames)} frames) ✅'
        if sensor is not None:
            fig = px.imshow(normalize_for_plot(sensor), origin='lower', color_continuous_scale='gray')
            fig.update_layout(title='Loaded sensor image', template='plotly_dark', height=600)
            return fig, sensor.tolist(), 'Sensor loaded ✅'
        f = go.Figure(); f.update_layout(title='No data', template='plotly_dark'); return f, None, 'No data loaded'
    except Exception:
        traceback.print_exc()
        f = go.Figure(); f.update_layout(title='Simulation error', template='plotly_dark'); return f, None, 'Simulation error'

@app.callback(
    Output('download_fits','data'),
    Input('dl_image','n_clicks'),
    State('store_sim','data')
)
def download_sim(n_clicks, sim_data):
    try:
        if n_clicks and sim_data is not None:
            arr = np.array(sim_data)
            b = fits_bytes(arr)
            return dcc.send_bytes(lambda f: f.write(b), filename='simulated_image.fits')
    except Exception:
        traceback.print_exc()
    return dash.no_update

@app.callback(
    Output('tab-content','children'),
    Input('main-tabs','value'),
    Input('ra-slider','value'),
    Input('dec-slider','value'),
    Input('mag-slider','value'),
    Input('planet-toggle','value'),
    Input('reset-btn','n_clicks')
)
def render_tab(tab, ra_range, dec_range, mag, planet_toggle, reset_clicks):
    if CATALOG0 is not None and isinstance(CATALOG0, list):
        catalog = CATALOG0
    else:
        N = 1200
        np.random.seed(0)
        catalog = [{"ra_deg":float(np.random.uniform(0,360)),
                    "dec_deg":float(np.random.uniform(-85,85)),
                    "brightness":float(np.random.uniform(8,16))} for _ in range(N)]
    planet_snapshot = PLANETS_SNAPSHOT if PLANETS_SNAPSHOT is not None else None
    show_planets = "YES" in (planet_toggle or [])
    if tab == "tab-sim":
        return html.Div([build_optical_layout()])
    else:
        fig2 = make_2d_sky_figure(catalog, planet_snapshot, ra_range, dec_range, mag, show_planets)
        fig3 = make_3d_globe_figure(catalog)
        return html.Div([
            dbc.Row([dbc.Col(dcc.Graph(figure=fig2, id='sky2d'), width=12)]),
            dbc.Row([dbc.Col(dcc.Graph(figure=fig3, id='globe3d'), width=12)])
        ])

# ----------------------------
# Run (auto-open browser)
# ----------------------------
if __name__ == "__main__":
    print(f"Starting {APP_TITLE}")
    print("Catalog (primary):", EXAMPLE_FILES['catalog'])
    print("Merged (optional):", EXAMPLE_FILES['merged_catalog'])
    try:
        webbrowser.open("http://127.0.0.1:8050/")
    except Exception:
        pass
    app.run(debug=True, port=8050, use_reloader=False)
