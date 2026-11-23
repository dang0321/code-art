"""snowflake.py - one snow crystal growing in a cold chamber, seen through a microscope as a
time-lapse: a tiny hexagonal plate, six arms, then side branches, grown on a hexagonal lattice
with the Gravner-Griffeath model (vapour diffuses, freezes onto the edge, attaches, melts back);
drawn with ASCII characters and rendered to a 1080x1920 video.

    python snowflake.py                      -> snowflake.mp4
    python snowflake.py out.mp4              -> custom output name
    python snowflake.py --still 10 a.png     -> one frame at t=10s
    python snowflake.py --stills 2,9,13 dir  -> several frames into a folder
"""
import hashlib, math, os, re, shutil, subprocess, sys, tempfile, time
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops

W, H, FPS, DUR = 1080, 1920, 30, 20.0
TAU = math.tau

CW, CH = 10, 16
COLS, ROWS = W // CW, 75
GY0, GH = 140, 75 * 16
HS = 2                                    # the picture is worked out at half resolution
HW, HH = W // HS, GH // HS
BG = (2, 3, 8)


def font(names, size):
    for name in names:
        for d in ("C:/Windows/Fonts", "/usr/share/fonts/truetype/dejavu", "/Library/Fonts"):
            p = os.path.join(d, name)
            if os.path.exists(p):
                return ImageFont.truetype(p, size)
    return ImageFont.load_default(size)


MONO = ["consola.ttf", "DejaVuSansMono.ttf", "Menlo.ttc"]
F_GRID, F_UI, F_CODE = font(MONO, 16), font(MONO, 26), font(MONO, 21)


def hash01(a, b, c=0.0):                  # cheap deterministic noise in [0, 1)
    v = np.sin(a * 12.9898 + b * 78.233 + c * 37.719) * 43758.5453
    return v - np.floor(v)


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


# --- chamber: the air the crystal grows in changes as it drifts, and so does its shape
#     each stage: (radius it starts at, as a share of the final radius), then the model's
#     constants rho, beta, alpha, theta, kappa, mu, gamma (Gravner & Griffeath 2008)
STAGES = [
    (0.00, "hexagonal plate", -2.0, 6, (0.38, 1.06, 0.35, 0.112, 0.001, 0.14, 0.0006)),
    (0.18, "six arms", -15.0, 22, (0.5, 1.3, 0.08, 0.025, 0.006, 0.07, 0.00005)),
]
R_FINAL = 105                             # lattice sites from the centre to the tip of an arm
SIGMA = 0.0003                            # a little noise in the vapour, so no two arms match exactly


# --- lattice: hexagonal sites stored row by row, every other row shifted by half a site
RD = R_FINAL + 22                         # beyond this radius the air is fresh: a vapour reservoir
NR = int(2 * RD / 0.8660254) + 8
NR += NR % 2
NC = int(2 * RD) + 8
SHAPE = (NR + 2, NC + 2)                  # one site of padding all round
I0, J0 = SHAPE[0] // 2, SHAPE[1] // 2
_ii, _jj = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
LX = _jj + 0.5 * (_ii & 1) - (J0 + 0.5 * (I0 & 1))                 # site centres, in site spacings
LY = (_ii - I0) * 0.8660254
LR = np.hypot(LX, LY)
EVEN = (_ii[1:-1, 1:-1] & 1) == 0
IN = np.s_[1:-1, 1:-1]


def nb6(Z):                               # sum over the six neighbours of every inner site
    return (Z[1:-1, 2:] + Z[1:-1, :-2] + Z[2:, 1:-1] + Z[:-2, 1:-1]
            + np.where(EVEN, Z[:-2, :-2] + Z[2:, :-2], Z[:-2, 2:] + Z[2:, 2:]))


def grow():
    """Run the crystal to full size; return the step each site froze at and its ice mass."""
    rng = np.random.default_rng(7)
    a = np.zeros(SHAPE, bool)             # attached: part of the crystal
    b = np.zeros(SHAPE, np.float32)       # boundary mass: the quasi-liquid skin
    c = np.zeros(SHAPE, np.float32)       # crystal mass: ice
    rho = STAGES[0][4][0]
    d = np.full(SHAPE, rho, np.float32)   # diffusive mass: vapour
    att = np.full(SHAPE, -1, np.int32)
    a[I0, J0], c[I0, J0], d[I0, J0], att[I0, J0] = True, 1.0, 0.0, 0
    fresh = LR > RD
    radius, stage, s, rad_log = 0.0, 0, 0, [0.0]
    while radius < R_FINAL:
        s += 1
        while stage + 1 < len(STAGES) and radius >= STAGES[stage + 1][0] * R_FINAL:
            stage += 1
        rho, beta, alpha, theta, kappa, mu, gamma = STAGES[stage][4]
        A = a.astype(np.float32)
        nA = np.zeros_like(A)
        nA[IN] = nb6(A)
        edge = ~a & (nA > 0)
        # --- diffusion: vapour averages with its neighbours; ice reflects it back
        dn = np.zeros_like(d)
        dn[IN] = (d[IN] + nb6(d) + nA[IN] * d[IN]) / 7.0
        dn[a] = 0
        dn[fresh] = rho
        d = dn
        # --- freezing: at the edge, vapour splits into skin (1 - kappa) and ice (kappa)
        fz = d * edge
        b += (1 - kappa) * fz
        c += kappa * fz
        d -= fz
        # --- attachment: a site joins when its skin is thick enough for how exposed it is
        near = np.zeros_like(d)
        near[IN] = nb6(d) + d[IN]
        join = edge & ((((nA == 1) | (nA == 2)) & (b >= beta))
                       | ((nA == 3) & ((b >= 1) | ((b >= alpha) & (near < theta))))
                       | (nA >= 4))
        if join.any():
            a |= join
            c[join] += b[join]
            b[join] = 0
            att[join] = s
            radius = max(radius, float(LR[join].max()))
        # --- melting: some skin and ice turn back into vapour
        e2 = edge & ~join
        mb, mc = mu * b * e2, gamma * c * e2
        b -= mb
        c -= mc
        d += mb + mc
        if SIGMA:
            d *= np.where(rng.random(d.shape) < 0.5, 1 - SIGMA, 1 + SIGMA).astype(np.float32)
        rad_log.append(radius)
    return att, c * a, np.array(rad_log)


def crystal():
    key = f"{STAGES}{R_FINAL}{SIGMA}"
    path = os.path.join(tempfile.gettempdir(), f"snowflake_{hashlib.md5(key.encode()).hexdigest()[:10]}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return z["att"], z["c"], z["rad"]
    t0 = time.time()
    att, c, rad = grow()
    print(f"grew the crystal in {len(rad) - 1} steps ({time.time() - t0:.0f}s)")
    np.savez_compressed(path, att=att, c=c, rad=rad)
    return att, c, rad


ATT, ICE, RAD = crystal()
STEPS = len(RAD) - 1
STAGE_STEP = [int(np.searchsorted(RAD, st[0] * R_FINAL)) for st in STAGES]
SITE_UM = 14.0                            # one lattice spacing, in micrometres
GROW_MIN = 21.0                           # minutes of real growth the whole run stands for


# --- microscope: the objective zooms out as the crystal grows, so the seed starts out large
CX, CY = HW / 2, HH / 2 + 4               # centre of the crystal, in half-resolution pixels
RS_FINAL = 236.0                          # tip radius on screen when it is done
T_DONE = 16.6
KNOTS = [(0.0, 0.012), (0.5, 0.025), (3.6, STAGES[1][0]), (15.4, 0.95), (16.0, 0.985), (T_DONE, 1.0)]


def growth(t):                            # -> share of the final radius, simulation step, zoom
    share = float(np.interp(t, *zip(*KNOTS)))
    step = int(min(np.searchsorted(RAD, share * R_FINAL), STEPS))
    zoom = RS_FINAL * share ** 0.6 / (share * R_FINAL)
    zoom *= 1 - 0.04 * smooth((t - T_DONE) / (DUR - T_DONE))
    return share, step, zoom


_v, _u = np.mgrid[0:HH, 0:HW].astype(np.float32)
SUBS = [(-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)]


ROOF_K = 26                               # how many sites in from the edge the faces keep rising


def roof(step):
    """Each branch is a low roof: thickest along its middle, so it has two sloping faces."""
    r = RAD[step] + 3
    i_lo, i_hi = max(I0 - int(r / 0.8660254) - 3, 0), min(I0 + int(r / 0.8660254) + 4, SHAPE[0])
    j_lo, j_hi = max(J0 - int(r) - 3, 0), min(J0 + int(r) + 4, SHAPE[1])
    at = ATT[i_lo:i_hi, j_lo:j_hi]
    D = np.where((at >= 0) & (at <= step), 99.0, 0.0).astype(np.float32)
    ev = ((np.arange(i_lo + 1, i_hi - 1) & 1) == 0)[:, None]
    for _ in range(ROOF_K):
        m = np.minimum.reduce([D[1:-1, 2:], D[1:-1, :-2], D[2:, 1:-1], D[:-2, 1:-1],
                               np.where(ev, np.minimum(D[:-2, :-2], D[2:, :-2]), np.minimum(D[:-2, 2:], D[2:, 2:]))])
        D[1:-1, 1:-1] = np.minimum(D[1:-1, 1:-1], m + 1)
    out = np.zeros(SHAPE, np.float32)
    out[i_lo:i_hi, j_lo:j_hi] = np.minimum(D, ROOF_K)
    return out


def sample(step, zoom):
    """Coverage, ice mass, freshness and roof height of the crystal under every pixel."""
    cov, ice, fresh, high = (np.zeros((HH, HW), np.float32) for _ in range(4))
    D = roof(step)
    for du, dv in SUBS:
        x = (_u + 0.5 + du - CX) / zoom
        y = (_v + 0.5 + dv - CY) / zoom
        best, bi, bj = None, None, None
        r0 = np.floor(y / 0.8660254).astype(np.int32)
        for rr in (r0, r0 + 1):                                     # nearest of two candidate rows
            row = rr + I0
            col = np.round(x + J0 + 0.5 * (I0 & 1) - 0.5 * (row & 1)).astype(np.int32)
            dx = x - (col + 0.5 * (row & 1) - J0 - 0.5 * (I0 & 1))
            dist = dx * dx + (y - rr * 0.8660254) ** 2
            if best is None:
                best, bi, bj = dist, row, col
            else:
                pick = dist < best
                best, bi, bj = np.where(pick, dist, best), np.where(pick, row, bi), np.where(pick, col, bj)
        ok = (bi >= 0) & (bi < SHAPE[0]) & (bj >= 0) & (bj < SHAPE[1])
        bi, bj = np.clip(bi, 0, SHAPE[0] - 1), np.clip(bj, 0, SHAPE[1] - 1)
        at = np.where(ok, ATT[bi, bj], -1)
        on = (at >= 0) & (at <= step)
        cov += on
        ice += np.where(on, ICE[bi, bj], 0)
        fresh += np.where(on, np.exp(-(step - at) / 140.0), 0)
        high += np.where(ok, D[bi, bj], 0)
    return cov / 4, ice / 4, fresh / 4, high / 4


def gauss_mat(n, sigma):
    x = np.arange(n, dtype=np.float32)
    m = np.exp(-0.5 * ((x[:, None] - x[None, :]) / sigma) ** 2)
    return m / m.sum(1, keepdims=True)


def blur(img, sigma):
    return gauss_mat(HH, sigma) @ img @ gauss_mat(HW, sigma).T


# --- light: dark-field, from below and to the side; ice shows up only where it bends light
ICE_COL = np.array([0.66, 0.84, 1.0])
RIM_COL = np.array([0.86, 0.95, 1.0])
FRESH_COL = np.array([0.55, 0.95, 1.0])


def light_dir(t):
    a = -2.3 + 0.07 * t + 2.4 * smooth((t - T_DONE + 0.4) / (DUR - T_DONE + 0.2))
    return math.cos(a), math.sin(a)


# --- specks: tiny droplets and ice dust drifting through the chamber, out of focus
_r = np.random.default_rng(3)
SPECK = dict(x=_r.uniform(0, HW, 70), y=_r.uniform(0, HH, 70), v=_r.uniform(2, 9, 70),
             w=_r.uniform(-0.6, 0.6, 70), b=_r.uniform(0.05, 0.35, 70) ** 1.6, ph=_r.uniform(0, TAU, 70))
BOKEH = [(_r.uniform(40, HW - 40), _r.uniform(60, HH - 60), _r.uniform(35, 80), _r.uniform(0.004, 0.01),
          np.array([0.5, 0.75, 1.0]) if k % 3 else np.array([1.0, 0.8, 0.55])) for k in range(6)]


def background(t):
    rr = np.hypot((_u - CX) / HW, (_v - CY) / HH)
    img = np.exp(-rr * 3.2)[..., None] * np.array([0.004, 0.012, 0.03])
    for bx, by, br, bb, bc in BOKEH:                                 # soft discs: droplets far from focus
        y = (by - 4 * t) % HH
        d = np.hypot(_u - bx, _v - y) / br
        img += (np.exp(-2 * d * d) * bb)[..., None] * bc
    xs = (SPECK["x"] + SPECK["w"] * t * 6) % HW
    ys = (SPECK["y"] - SPECK["v"] * t) % HH
    tw = SPECK["b"] * (0.7 + 0.3 * np.sin(SPECK["ph"] + t * 2.1))
    for x, y, b in zip(xs, ys, tw):
        i0, j0 = int(y), int(x)
        sl = np.s_[max(i0 - 2, 0):i0 + 3, max(j0 - 2, 0):j0 + 3]
        d2 = (_v[sl] - y) ** 2 + (_u[sl] - x) ** 2
        img[sl] += (np.exp(-d2 / 1.2) * b)[..., None] * np.array([0.8, 0.9, 1.0])
    return img


# --- shade: thickness steps and edges glow, flat ice stays dim, new ice glints
def shade(t, step, zoom):
    cov, ice, fresh, high = sample(step, zoom)
    sig = max(0.45 * zoom, 0.7)
    cb = blur(cov, sig)
    hb = blur(ice, sig)
    inside = blur(cov, sig * 2.2)
    th = hb / np.maximum(cb, 0.08)                                   # thickness of the ice itself
    gy, gx = np.gradient(cb)
    gm = np.hypot(gx, gy)
    rim = np.clip(gm * sig * 2.4, 0, 1.2)
    sr = max(sig, 1.8)                                               # ribs: only steps wider than a few sites
    thr = blur(ice, sr) / np.maximum(blur(cov, sr), 0.08)
    ty, tx = np.gradient(thr)
    ridge = np.clip(np.hypot(tx, ty) * sr * 2.6, 0, 1.2) * smooth((inside - 0.7) / 0.3)
    lx, ly = light_dir(t)
    hy, hx = np.gradient(hb)
    relief = (hx * lx + hy * ly) * sig * 3.0
    facing = np.clip(-(gx * lx + gy * ly) / np.maximum(gm, 1e-6), 0, 1)
    glint = rim * facing ** 12 * 1.5                                 # facets turned to the light flash
    H = blur(high, sig)
    sy, sx = np.gradient(H)
    sx, sy = sx * zoom, sy * zoom                                    # slope of the faces, per site
    sm = np.hypot(sx, sy)
    tilt = np.clip(sm, 0, 1.2) * smooth((inside - 0.5) / 0.4)
    face = tilt * np.clip(0.1 - 0.9 * (sx * lx + sy * ly) / np.maximum(sm, 1e-6), 0, 1) ** 1.5
    lap = (np.roll(H, 1, 0) + np.roll(H, -1, 0) + np.roll(H, 1, 1) + np.roll(H, -1, 1) - 4 * H) * zoom
    rib = np.clip(blur(np.clip((-lap - 0.35) * 2.0, 0, 1), 0.7) * 2.2, 0, 1.3)   # where two faces meet
    body = cb * (0.05 + 0.06 * np.clip(th - 0.5, 0, 1.5))
    lum = body + 0.85 * rim + 0.15 * ridge + 0.22 * face + 1.0 * rib + np.clip(relief, -0.3, 0.8) * 0.15 * cb
    tint = 0.5 + 0.5 * np.cos(TAU * (thr * 0.9 + np.array([0.0, 0.33, 0.67])[:, None, None]))
    col = ICE_COL[:, None, None] * (0.8 + 0.2 * tint)
    mix = np.clip(rim, 0, 1)[None]
    col = col * (1 - mix) + RIM_COL[:, None, None] * mix
    img = np.clip(lum, 0, None)[None] * col + glint[None]
    fr = blur(fresh, sig * 1.3) * smooth(1 - t / T_DONE + 0.05)
    img = img + FRESH_COL[:, None, None] * (fr * 0.55)[None]
    return img.transpose(1, 2, 0), rim, rib


# --- chars: bright ice gets dense characters, thin straight edges get a stroke along them
RAMP = np.array(list(" .,:;-=+*o%#@"))
STROKES = np.array(["-", "\\", "|", "/"])                          # lines at 0, 58, 90 and 122 degrees
STROKE_DEG = np.array([0.0, 58.0, 90.0, 122.0])
MASS_AT = np.cumsum(np.bincount(np.where(ATT >= 0, ATT, 0).ravel(), weights=(ICE * (ATT >= 0)).ravel(),
                                minlength=STEPS + 1))
UG_PER_UNIT = 0.8660254 * (SITE_UM * 1e-3) ** 2 * 0.012 * 0.917 * 1000      # ice mass per unit, in µg


def cells(img, how="mean"):
    return getattr(img.reshape(ROWS, CH // HS, COLS, CW // HS, *img.shape[2:]), how)((1, 3))


def orient(field):                       # per cell: a stroke along the line in it, and how line-like it is
    gy, gx = np.gradient(field)
    jxx, jyy, jxy = cells(gx * gx), cells(gy * gy), cells(gx * gy)
    coh = np.hypot(jxx - jyy, 2 * jxy) / (jxx + jyy + 1e-7)
    ang = (np.degrees(0.5 * np.arctan2(2 * jxy, jxx - jyy)) + 90) % 180     # along the line, y down
    dd = np.abs((ang[..., None] - STROKE_DEG + 90) % 180 - 90)
    return STROKES[dd.argmin(-1)], coh


def glyphs(img, rim, rib):
    m3 = cells(img)
    m = m3.max(-1)
    lum = 1 - np.exp(-2.2 * m)
    ch = np.where(lum < 0.06, " ", np.where(lum < 0.15, ".", np.where(lum < 0.24, ":",
                  RAMP[np.clip((lum * 13).astype(int), 0, 12)])))       # the faces stay sparse
    s_rim, c_rim = orient(rim)
    s_rib, c_rib = orient(rib)
    rim_c, rib_c = cells(rim), cells(rib, "max")
    on_rim = (rim_c > 0.22) & (c_rim > 0.45) & (lum < 0.75)
    on_rib = (rib_c > 0.4) & (c_rib > 0.4) & (rib_c > rim_c * 1.5) & (lum < 0.75)
    ch = np.where(on_rib, s_rib, np.where(on_rim, s_rim, ch))
    rgb = (np.clip(m3 / np.maximum(m, 1e-9)[..., None] * (0.25 + 0.85 * lum)[..., None]
                   + np.clip(lum - 0.85, 0, 1)[..., None] * 2, 0, 1) * 255).astype(int)
    return ch, rgb


def stage_label(share, t):
    if t > T_DONE:
        return "fully grown"
    return "hexagonal plate" if share < STAGES[1][0] else "six arms" if share < 0.3 else "side branches"


def conditions(step):
    w = np.array([1.0] + [smooth((step - STAGE_STEP[k]) / 400 + 0.5) for k in range(1, len(STAGES))])
    w = np.append(w[:-1] - w[1:], w[-1])
    return (w * [s[2] for s in STAGES]).sum(), (w * [s[3] for s in STAGES]).sum()


def frame(t):
    share, step, zoom = growth(t)
    ice_img, rim, rib = shade(t, step, zoom)
    img = background(t) + ice_img * smooth(t / 0.7) * 1.2
    ch, rgb = glyphs(img, rim, rib)
    under = Image.fromarray((np.clip(1 - np.exp(-2.4 * img), 0, 1) * 255 * 0.32).astype(np.uint8)).resize((W, GH), Image.BILINEAR)
    txt = Image.new("RGB", (W, GH), (0, 0, 0))
    d = ImageDraw.Draw(txt)
    for r, c in np.argwhere(ch != " ").tolist():
        d.text((c * CW, r * CH), ch[r, c], font=F_GRID, fill=tuple(rgb[r, c].tolist()))
    glow = txt.filter(ImageFilter.GaussianBlur(2)).point(lambda v: int(v * 0.5))
    halo = txt.filter(ImageFilter.GaussianBlur(14)).point(lambda v: min(255, int(max(0, v - 60) * 0.9)))
    out_img = Image.new("RGB", (W, GH), BG)
    for layer in (under, halo, glow, txt):
        out_img = ImageChops.add(out_img, layer)
    out = Image.new("RGB", (W, H), BG)
    out.paste(out_img, (0, GY0))
    d = ImageDraw.Draw(out)
    temp, ssat = conditions(step)
    dia = 2 * RAD[step] * SITE_UM / 1000
    mass = MASS_AT[min(step, len(MASS_AT) - 1)] * UG_PER_UNIT
    sec = int(step / STEPS * GROW_MIN * 60)
    d.text((36, 34), "snowflake.py", font=F_UI, fill=(204, 212, 222))
    d.text((36, 66), f"{temp:+5.1f} °C   supersaturation {ssat:4.1f}%   {stage_label(share, t)}",
           font=F_UI, fill=(128, 140, 156))
    d.text((36, 98), f"diameter {dia:4.2f} mm   ice {mass:5.2f} µg   grown {sec // 60:02d}:{sec % 60:02d}",
           font=F_UI, fill=(128, 140, 156))
    ts = f"t {t:05.2f}s"
    d.text((W - 36 - F_UI.getlength(ts), 34), ts, font=F_UI, fill=(128, 140, 156))
    code_panel(t, out)
    return out


# code panel: this very file scrolls by, following what is happening on screen
SRC = open(__file__, encoding="utf-8").read().splitlines()
TOK = re.compile(r"(?P<com>#.*)|(?P<str>\"[^\"]*\"|'[^']*')"
                 r"|(?P<kw>\b(?:def|for|in|if|elif|else|return|import|from|as|while|"
                 r"with|lambda|not|and|or|None|True|False|class)\b)|(?P<num>\b\d+(?:\.\d+)?\b)")
TOK_COL = {"com": (100, 116, 134), "str": (152, 195, 121), "kw": (120, 190, 240),
           "num": (209, 154, 102), None: (170, 180, 194)}


def tokens(line):
    out, pos = [], 0
    for m in TOK.finditer(line):
        out += [(line[pos:m.start()], None), (m.group(), m.lastgroup)]
        pos = m.end()
    return out + [(line[pos:], None)]


SRC_TOK = [tokens(l) for l in SRC]
PHASES = [(0.0, "chamber"), (1.4, "lattice"), (2.8, "diffusion"), (4.2, "freezing"), (5.6, "attachment"),
          (7.4, "melting"), (8.6, "microscope"), (10.0, "light"), (11.2, "shade"), (12.6, "specks"),
          (14.0, "chamber"), (15.4, "chars"), (17.2, "render"), (18.8, "encode"), (DUR, "encode")]
KEYS = [(t, next(i for i, l in enumerate(SRC) if l.strip().startswith("# --- " + name)))
        for t, name in PHASES]


def code_pos(t):
    for (t0, l0), (t1, l1) in zip(KEYS, KEYS[1:]):
        if t < t1 or t1 == KEYS[-1][0]:
            span = max(abs(l1 - l0) - 3, 0)
            drift = l0 + min((t - t0) * 2.5, span)
            if t > t1 - 0.5 and l1 != l0:
                end = l0 + min((t1 - 0.5 - t0) * 2.5, span)
                s = (t - (t1 - 0.5)) / 0.5
                return end + (l1 - end) * s * s * (3 - 2 * s)
            return drift
    return KEYS[-1][1]


PX0, PY0, PW, PH_ = 30, 1365, W - 60, 525
LH, CODE_TOP = 27, 52
CHAR_W = F_CODE.getlength("M")


def code_panel(t, img):
    panel = Image.new("RGB", (PW, PH_), (14, 18, 26))
    d = ImageDraw.Draw(panel)
    pos = code_pos(t)
    top = pos - 6
    cur = int(round(pos))
    for li in range(int(math.floor(top)), int(top) + 22):
        if not 0 <= li < len(SRC):
            continue
        yy = CODE_TOP + (li - top) * LH
        if yy < CODE_TOP - LH or yy > PH_:
            continue
        if li == cur:
            d.rectangle((0, yy - 2, PW, yy + LH - 3), fill=(28, 36, 52))
        d.text((12, yy), f"{li + 1:>3}", font=F_CODE, fill=(78, 88, 104))
        x = 70
        for seg, kind in SRC_TOK[li]:
            seg = seg[: max(0, int((PW - 20 - x) / CHAR_W))]
            if seg:
                d.text((x, yy), seg, font=F_CODE, fill=TOK_COL[kind])
                x += len(seg) * CHAR_W
    d.rectangle((0, 0, PW, 52), fill=(14, 18, 26))           # title bar over the code
    for i, c in enumerate([(255, 95, 87), (254, 188, 46), (40, 200, 64)]):
        d.ellipse((20 + i * 26, 16, 36 + i * 26, 32), fill=c)
    d.text((110, 11), "snowflake.py", font=F_UI, fill=(204, 212, 222))
    d.ellipse((PW - 250, 18, PW - 236, 32), fill=(40, 200, 64))
    d.text((PW - 228, 11), "running  30 fps", font=F_UI, fill=(144, 154, 168))
    d.line((0, 46, PW, 46), fill=(34, 42, 58), width=2)
    mask = Image.new("L", (PW, PH_), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, PW - 1, PH_ - 1), 18, fill=255)
    img.paste(panel, (PX0, PY0), mask)
    ImageDraw.Draw(img).rounded_rectangle((PX0, PY0, PX0 + PW - 1, PY0 + PH_ - 1), 18,
                                         outline=(38, 48, 66), width=2)


# --- render: the frame above, piped straight to the encoder below
# --- encode: pipe raw frames into ffmpeg
def encode(out):
    ff = shutil.which("ffmpeg") or "ffmpeg"
    proc = subprocess.Popen(
        [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium",
         "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out],
        stdin=subprocess.PIPE)
    total = int(DUR * FPS)
    for f in range(total):
        proc.stdin.write(frame(f / FPS).tobytes())
        if f % FPS == 0:
            print(f"\rrendering {f / FPS:4.0f}s / {DUR:.0f}s", end="", flush=True)
    proc.stdin.close()
    proc.wait()
    print(f"\rdone -> {out}            ")


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--still":
        frame(float(sys.argv[2])).save(sys.argv[3])
    elif len(sys.argv) >= 4 and sys.argv[1] == "--stills":
        os.makedirs(sys.argv[3], exist_ok=True)
        for ts in sys.argv[2].split(","):
            frame(float(ts)).save(os.path.join(sys.argv[3], f"t{float(ts):05.2f}.png"))
    else:
        encode(sys.argv[1] if len(sys.argv) > 1 else "snowflake.mp4")
