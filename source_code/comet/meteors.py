"""meteors.py - a meteor shower over a mountain lake: the Milky Way overhead, streaks shooting
out from one point in the sky, a fireball that lights the valley and leaves a glowing train,
all mirrored in still water; drawn with ASCII characters and rendered to a 1080x1920 video.

    python meteors.py                      -> meteors.mp4
    python meteors.py out.mp4              -> custom output name
    python meteors.py --still 10 a.png     -> one frame at t=10s
    python meteors.py --stills 2,9,13 dir  -> several frames into a folder
"""
import math, os, re, shutil, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops

W, H, FPS, DUR = 1080, 1920, 30, 20.0
TAU = math.tau
rng = np.random.default_rng(53)

CW, CH = 10, 16
COLS, ROWS = W // CW, 75
GY0, GH = 140, 75 * 16
SHORE = 52                                # first row of the lake
Y_SHORE = SHORE * CH
BG = (2, 3, 7)


def font(names, size):
    for name in names:
        for d in ("C:/Windows/Fonts", "/usr/share/fonts/truetype/dejavu", "/Library/Fonts"):
            p = os.path.join(d, name)
            if os.path.exists(p):
                return ImageFont.truetype(p, size)
    return ImageFont.load_default(size)


MONO = ["consola.ttf", "DejaVuSansMono.ttf", "Menlo.ttc"]
F_GRID, F_UI, F_CODE = font(MONO, 16), font(MONO, 26), font(MONO, 21)
F_SMALL, F_HEAD = font(MONO, 12), font(MONO, 22)


def hash01(a, b, c=0.0):                  # cheap deterministic noise in [0, 1)
    v = np.sin(a * 12.9898 + b * 78.233 + c * 37.719) * 43758.5453
    return v - np.floor(v)


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def vnoise(x, y):
    xi, yi = np.floor(x), np.floor(y)
    u, v = smooth(x - xi), smooth(y - yi)
    a, b = hash01(xi, yi), hash01(xi + 1, yi)
    c, d = hash01(xi, yi + 1), hash01(xi + 1, yi + 1)
    return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v


def fbm(x, y, octaves=4):
    s, amp, norm = 0.0, 1.0, 0.0
    for i in range(octaves):
        s = s + amp * vnoise(x * 2 ** i + 17 * i, y * 2 ** i - 9 * i)
        norm += amp
        amp *= 0.5
    return s / norm


_cc, _rr = np.meshgrid(np.arange(COLS), np.arange(ROWS))


# --- sky: a moonless night far from town; the only light is starlight and a faint airglow
def sky_glow(py):                         # per pixel row: airglow and a far town on the horizon
    h = np.clip((Y_SHORE - py) / Y_SHORE, 0, 1)
    return np.exp(-h / 0.1)[..., None] * np.array([7.0, 13.0, 9.0])


# --- milkyway: the galaxy is countless faint stars plus a glow, split by dark dust
MW_P0, MW_U = np.array([-80.0, 900.0]), np.array([1300.0, -1150.0]) / math.hypot(1300, 1150)


def mw_density(x, y):
    q = np.stack([x, y], -1) - MW_P0
    s = q @ MW_U
    d = q[..., 0] * MW_U[1] - q[..., 1] * MW_U[0]
    band = np.exp(-(d / 170) ** 2) * (0.35 + 0.65 * fbm(s / 110, d / 55, 5)) ** 1.6      # clumpy star clouds
    core = 1 + 1.4 * np.exp(-((s - 330) / 320) ** 2)
    lane = 18 + 38 * np.sin(s / 210) + 20 * (fbm(s / 90, 3.0, 3) - 0.5)                    # the dark rift
    rift = 1 - 0.85 * np.exp(-((d - lane) / 32) ** 2) * smooth((900 - s) / 400)
    warm = np.exp(-((s - 330) / 380) ** 2)
    return band * core * rift, warm


def make_stars():
    r = np.random.default_rng(7)
    xs, ys, bs, cs = [], [], [], []
    n = 2400                                                     # the ordinary field, all over the sky
    xs.append(r.uniform(0, W, n)); ys.append(r.uniform(0, Y_SHORE, n))
    mag = 1.0 + 5.6 * r.random(n) ** 0.3
    bs.append(1.0 * 10 ** (-0.4 * (mag - 1.0)))
    temp = r.random(n)
    cs.append(np.where(temp[:, None] < 0.25, [0.74, 0.84, 1.0], np.where(temp[:, None] < 0.75, [1.0, 0.97, 0.92],
              np.where(temp[:, None] < 0.92, [1.0, 0.86, 0.68], [1.0, 0.72, 0.5]))))
    x, y = r.uniform(0, W, 90000), r.uniform(0, Y_SHORE, 90000)   # the galaxy: faint stars, thinned by its density
    dens, warm = mw_density(x, y)
    keep = r.random(len(x)) < 0.22 * dens
    xs.append(x[keep]); ys.append(y[keep])
    k = int(keep.sum())
    bs.append(0.018 + 0.05 * r.random(k) ** 3)
    w = warm[keep][:, None]
    cs.append(np.array([0.98, 0.86, 0.7]) * w + np.array([0.78, 0.84, 1.0]) * (1 - w))
    b = np.concatenate(bs)
    return dict(x=np.concatenate(xs), y=np.concatenate(ys), b=b, col=np.concatenate(cs).astype(float),
                ph=r.uniform(0, TAU, len(b)), w=r.uniform(6, 18, len(b)))


STARS = make_stars()


# --- mountains: black against the stars, with no outline at all; only the fireball shows their faces
def make_land():
    r = np.random.default_rng(3)
    x = np.arange(W)
    h = np.zeros(W)
    for cx, hh, ww in ((80, 95, 150), (270, 160, 170), (450, 115, 130), (680, 245, 210), (900, 175, 160), (1040, 125, 110)):
        h = np.maximum(h, hh * np.clip(1 - np.abs(x - cx) / ww, 0, None) ** 1.1)
    h += 7 * np.sin(x / 17) + 4 * np.sin(x / 6.3) + 3 * (fbm(x / 9, 1.0, 3) - 0.5)
    pines = 14 * np.clip(np.sin(x * 0.7) * 0.5 + 0.5, 0, 1) * (hash01(np.floor(x / 9), 3.0) > 0.3) + 6
    ridge = Y_SHORE - np.maximum(h, pines)                       # pixel row of the skyline at each x
    return ridge


RIDGE = make_land()
LANTERN = (230.0, Y_SHORE - 6.0)


# --- radiant: every meteor of the shower seems to fly out from this one point, here in the frame
RADIANT = np.array([300.0, 200.0])


def ridge_at(x):
    return RIDGE[np.clip(np.asarray(x, int), 0, W - 1)]


def make_meteors():
    r = np.random.default_rng(11)
    out = []
    ts = np.arange(0.0, DUR, 1 / 400)
    rate = 10 + 80 * np.exp(-((ts - 12.0) / 5.5) ** 2)           # a storm, building to a peak
    for t0 in ts[r.random(len(ts)) < rate / 400]:
        for _ in range(60):                                    # find a path that starts in open sky
            th = r.uniform(0, TAU)
            dvec = np.array([math.cos(th), math.sin(th)])
            r0 = 40 + 820 * r.random() ** 1.3
            L = (0.2 + 0.8 * min(r0 / 700, 1)) * r.uniform(120, 420)   # near the radiant they come at us: short
            p0 = RADIANT + dvec * r0
            if 0 < p0[0] < W and 0 < p0[1] < ridge_at(p0[0]) - 10:
                break
        kind = r.random()
        col = [0.88, 0.93, 1.0] if kind < 0.6 else [0.62, 1.0, 0.74] if kind < 0.85 else [1.0, 0.84, 0.6]
        v = r.uniform(650, 1400)
        out.append(dict(t0=t0, p0=p0, d=dvec, L=L, dur=L / v, B=0.9 + 2.4 * r.random() ** 3,
                        col=np.array(col), big=False))
    # --- fireball: three big ones, brighter than Venus, each breaking up at the end
    for t0, th, r0, L in ((5.6, 0.95, 260, 520), (11.4, 0.42, 330, 600), (16.3, 1.75, 300, 460)):
        dvec = np.array([math.cos(th), math.sin(th)])
        out.append(dict(t0=t0, p0=RADIANT + dvec * r0, d=dvec, L=L, dur=L / 480, B=5.5,
                        col=np.array([0.66, 1.0, 0.74]), big=True))
    return out


METEORS = make_meteors()
FIREBALLS = [m for m in METEORS if m["big"]]


def brightness(m, u):                     # rises, burns, flickers, flares as it breaks up, gone
    u = np.clip(u, 0, 1)
    return m["B"] * np.sin(np.pi * u) ** 0.6 * (1 + 1.4 * np.exp(-((u - 0.83) / 0.05) ** 2)) \
        * (0.85 + 0.15 * np.sin(97 * u * m["dur"] + m["t0"]))


def flash(t):                             # how much the fireballs light the whole valley
    f = 0.0
    for m in FIREBALLS:
        u = (t - m["t0"]) / m["dur"]
        if 0 < u < 1.2:
            f += float(brightness(m, min(u, 1.0)) / m["B"] * (1 if u <= 1 else math.exp(-(u - 1) / 0.05)))
    return f


def put(d, x, y, g, col, font_=None):     # a character in the sky, and its double in the lake
    font_ = font_ or F_GRID
    if y < ridge_at(x):
        d.text((x - 3, y - 8), g, font=font_, fill=tuple(int(c * 255) for c in np.clip(col, 0, 1)))
        yr = 2 * Y_SHORE - y + 4
        if yr < GH - 6:
            k = (yr - Y_SHORE) / (GH - Y_SHORE)
            d.text((x - 3 + (2 + 8 * k) * math.sin(yr * 0.07), yr - 8), g, font=font_,
                   fill=tuple(int(c * 255 * (0.4 - 0.18 * k)) for c in np.clip(col, 0, 1)))


# --- meteors: a thin streak of light, brightest at the head, gone in a fraction of a second
def meteors(t, d, glow):
    for m in METEORS:
        u = (t - m["t0"]) / m["dur"]
        linger = 0.9 if (m["B"] > 1.2 and not m["big"]) else 0.25   # bright ones leave a short-lived glow
        if not 0 < u < 1.0 + linger / m["dur"]:
            continue
        if u <= 1:
            b = float(brightness(m, u))
            head = m["p0"] + m["d"] * m["L"] * u
            tail = min(0.5 * m["L"], m["L"] * u)
            x = np.arange(0, tail, 3.0)
            k = np.exp(-x / (0.3 * tail + 1))
        else:                                                   # the ionised path fades
            age = (u - 1) * m["dur"]
            b = 0.3 * m["B"] * math.exp(-age / (0.07 if linger < 0.5 else 0.3))
            head = m["p0"] + m["d"] * m["L"]
            x = np.arange(0, m["L"] * 0.7, 3.0)
            k = np.exp(-x / (0.3 * m["L"]))
        pts = head - m["d"] * x[:, None]
        step = 2 if b > 0.6 else 3
        for (px, py), kk in zip(pts[::step], k[::step]):
            v = m["col"] * b * kk * 0.95
            if v.max() > 0.05:
                put(d, px, py, "·" if kk > 0.3 else ".", v)
        line = [tuple(p) for p in pts[:: max(1, len(pts) // 12)] if p[1] < ridge_at(p[0])]
        if len(line) > 1:
            glow.line(line, fill=tuple(int(c * 255) for c in np.clip(m["col"] * b * 0.45, 0, 1)), width=3)
        if u <= 1:
            put(d, head[0] - 2, head[1] - 4, "•", np.array([1.0, 1.0, 0.95]) * b, F_HEAD if m["big"] or b > 1.5 else None)
            if m["big"] and u > 0.82:                            # fragments peel away from the breaking head
                for j in range(6):
                    off = m["d"] * (u - 0.82) * m["L"] * (0.12 * j) + np.array([(j - 2.5) * 4, j * 2.0]) * (u - 0.82) * 8
                    put(d, head[0] - off[0], head[1] - off[1], "·", m["col"] * b * 0.5)


# --- train: a fireball's trail keeps glowing for seconds, twisted by winds 90 km up
def train(t, d, glow):
    for m in FIREBALLS:
        age = t - (m["t0"] + m["dur"])
        if not 0 < age < 9:
            continue
        X = np.arange(0, m["L"] * 0.95, 4.0)
        s = X / m["L"]
        P = m["p0"] + m["d"] * X[:, None]
        P = P + np.stack([26 * np.sin(s * 9 + m["t0"]) + 12 * np.sin(s * 23), 9 * np.cos(s * 7)], -1) * age / 4
        fade = math.exp(-age / 3.0) * smooth(age / 0.3)
        mix = math.exp(-age / 2.2)
        col = np.array([0.5, 1.0, 0.66]) * mix + np.array([0.9, 0.62, 0.42]) * (1 - mix)
        vis = [tuple(p) for p in P[::6] if p[1] < ridge_at(p[0])]
        if len(vis) > 1:
            glow.line(vis, fill=tuple(int(c * 255 * 0.3 * fade) for c in col), width=int(4 + 3 * age))
        for (px, py), ss in zip(P[::5], s[::5]):
            v = fade * (0.3 + 0.25 * math.sin(ss * 40 + age))
            if v > 0.04:
                put(d, px, py, "~" if v > 0.2 else ".", col * v)


# --- lake: still water mirrors everything, broken up by slow ripples
def mirror_rows(t):
    rows = np.arange(SHORE, ROWS)
    src = np.clip(2 * SHORE - rows - 1, 0, SHORE - 1)
    depth = (rows - SHORE) / (ROWS - SHORE)
    wob = np.round((0.4 + 1.6 * depth)[:, None] * np.sin(t * 1.3 + rows[:, None] * 1.1)
                   + 0.6 * np.sin(t * 2.0 + _cc[SHORE:] * 0.3 + rows[:, None])).astype(int)
    return rows, src, depth, wob


# --- chars: only what is actually there; each star sits exactly where it is, the dark stays dark
STAR_LUM_MIN = 0.03


def stars(t, d):
    tw = 1 + 0.3 * (0.3 + 0.7 * STARS["y"] / Y_SHORE) * np.sin(STARS["w"] * t + STARS["ph"])   # worse twinkling low down
    above = STARS["y"] < RIDGE[np.clip(STARS["x"].astype(int), 0, W - 1)] - 2
    lum = 1 - np.exp(-2.2 * STARS["b"] * tw)
    pick = np.nonzero(above & (lum > STAR_LUM_MIN))[0]
    for i in pick:
        L = lum[i]
        g, f = ("*", F_GRID) if L > 0.7 else ("+", F_GRID) if L > 0.42 else ("\u00b7", F_GRID) if L > 0.16 else (".", F_SMALL)
        c = STARS["col"][i] * (0.3 + 0.9 * L)
        d.text((STARS["x"][i] - 3, STARS["y"][i] - (8 if f is F_GRID else 7)), g, font=f, fill=tuple(int(v * 255) for v in np.clip(c, 0, 1)))
    # --- lake: the same stars upside down, dimmer, shivered by slow ripples
    refl = pick[lum[pick] > 0.1]
    yr = 2 * Y_SHORE - STARS["y"][refl] + 4
    keep = yr < GH - 4
    for i, y in zip(refl[keep], yr[keep]):
        depth = (y - Y_SHORE) / (GH - Y_SHORE)
        if hash01(i, math.floor(t * 5 + i % 7)) < 0.18:
            continue
        x = STARS["x"][i] + (2 + 10 * depth) * math.sin(t * 1.4 + y * 0.07) + 3 * math.sin(t * 2.3 + STARS["x"][i] * 0.03)
        c = STARS["col"][i] * lum[i] * (0.45 - 0.2 * depth)
        d.text((x - 3, y - 8), "-" if depth > 0.4 and lum[i] > 0.3 else "\u00b7", font=F_GRID,
               fill=tuple(int(v * 255) for v in np.clip(c, 0, 1)))


def frame(t):
    fl = flash(t)
    txt = Image.new("RGB", (W, GH), (0, 0, 0))
    glow_l = Image.new("RGB", (W, GH), (0, 0, 0))
    d, g = ImageDraw.Draw(txt), ImageDraw.Draw(glow_l)
    stars(t, d)
    meteors(t, d, g)
    train(t, d, g)
    # a lantern by the tent on the shore, and its long shimmer on the water
    lx, ly = LANTERN
    d.text((lx - 4, ly - 10), "\u2022", font=F_GRID, fill=(255, 176, 92))
    g.ellipse((lx - 14, ly - 14, lx + 14, ly + 14), fill=(120, 70, 30))
    for k in range(24):
        yy = Y_SHORE + 8 + k * 9
        if hash01(k, math.floor(t * 6)) > 0.35:
            sh = 3 * math.sin(t * 2.2 + k * 0.9)
            v = math.exp(-k / 10)
            d.text((lx - 3 + sh, yy - 8), "|" if v > 0.5 else "'", font=F_GRID, fill=(int(220 * v), int(140 * v), int(70 * v)))
    # the land: black, and the stars stop at its edge; the fireball shows its slopes for a moment
    land = np.zeros((GH, W), bool)
    yy = np.arange(GH)[:, None]
    land[:Y_SHORE] = (yy[:Y_SHORE] >= RIDGE[None, :])
    if fl > 0.05:
        for c in range(0, COLS):
            top = int(RIDGE[c * CW + 5] // CH) + 1
            for r in range(top, SHORE):
                if hash01(c, r) < 0.35:
                    v = int(110 * fl * (0.4 + 0.6 * hash01(r, c)))
                    d.text((c * CW, r * CH), ".", font=F_GRID, fill=(int(v * 0.7), v, int(v * 0.8)))
    # glows under the characters: the galaxy's diffuse light, the airglow, the lake reflecting both
    gx, gy = np.meshgrid(np.arange(0, W, 4) + 2.0, np.arange(0, GH, 4) + 2.0)
    dens, warm = mw_density(gx, gy)
    mw = (dens[..., None] * (np.array([30.0, 26, 22]) * warm[..., None] + np.array([20.0, 24, 32]) * (1 - warm[..., None])))
    base = mw + sky_glow(gy) + fl * np.array([14.0, 22, 16])
    sky = gy < Y_SHORE
    refl = base[np.clip((2 * Y_SHORE - gy).astype(int) // 4, 0, base.shape[0] - 1), (gx // 4).astype(int)] * 0.35
    base = np.where(sky[..., None], base, refl)
    under = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).resize((W, GH), Image.BILINEAR).filter(ImageFilter.GaussianBlur(8))
    glow_l = glow_l.filter(ImageFilter.GaussianBlur(3))
    halo = txt.filter(ImageFilter.GaussianBlur(12)).point(lambda v: min(255, int(v * (0.6 + 0.8 * fl))))
    soft = txt.filter(ImageFilter.GaussianBlur(1.5)).point(lambda v: int(v * 0.5))
    img = Image.new("RGB", (W, GH), BG)
    for layer in (under, glow_l, halo, soft, txt):
        img = ImageChops.add(img, layer)
    lm = Image.fromarray((land * 255).astype(np.uint8))
    img.paste(Image.new("RGB", (W, GH), BG), (0, 0), lm)            # the silhouette cuts everything off
    out = Image.new("RGB", (W, H), BG)
    out.paste(img, (0, GY0))
    d = ImageDraw.Draw(out)
    seen = sum(1 for mm in METEORS if mm["t0"] <= t)
    d.text((36, 34), "meteors.py", font=F_UI, fill=(206, 208, 222))
    d.text((36, 66), f"Perseid storm  ZHR 9,000   seen {seen:3d}", font=F_UI, fill=(128, 134, 156))
    peak = "fireball  -8 mag" if fl > 0.05 else "sky  21.7 mag/arcsec\u00b2"
    d.text((36, 98), f"{peak}   02:{14 + int(t // 60):02d}:{int(t % 60):02d}", font=F_UI, fill=(128, 134, 156))
    ts = f"t {t:05.2f}s"
    d.text((W - 36 - F_UI.getlength(ts), 34), ts, font=F_UI, fill=(128, 134, 156))
    code_panel(t, out)
    return out


# code panel: this very file scrolls by, following what is happening on screen
SRC = open(__file__, encoding="utf-8").read().splitlines()
TOK = re.compile(r"(?P<com>#.*)|(?P<str>\"[^\"]*\"|'[^']*')"
                 r"|(?P<kw>\b(?:def|for|in|if|elif|else|return|import|from|as|while|"
                 r"with|lambda|not|and|or|None|True|False|class)\b)|(?P<num>\b\d+(?:\.\d+)?\b)")
TOK_COL = {"com": (104, 112, 136), "str": (152, 195, 121), "kw": (160, 150, 240),
           "num": (209, 154, 102), None: (172, 178, 194)}


def tokens(line):
    out, pos = [], 0
    for m in TOK.finditer(line):
        out += [(line[pos:m.start()], None), (m.group(), m.lastgroup)]
        pos = m.end()
    return out + [(line[pos:], None)]


SRC_TOK = [tokens(l) for l in SRC]
PHASES = [(0.0, "sky"), (1.2, "milkyway"), (2.8, "mountains"), (4.0, "radiant"), (5.4, "fireball"),
          (7.0, "meteors"), (9.0, "train"), (11.0, "lake"), (13.6, "chars"), (16.4, "render"),
          (19.0, "encode"), (DUR, "encode")]
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
    panel = Image.new("RGB", (PW, PH_), (14, 16, 26))
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
            d.rectangle((0, yy - 2, PW, yy + LH - 3), fill=(30, 33, 54))
        d.text((12, yy), f"{li + 1:>3}", font=F_CODE, fill=(78, 84, 106))
        x = 70
        for seg, kind in SRC_TOK[li]:
            seg = seg[: max(0, int((PW - 20 - x) / CHAR_W))]
            if seg:
                d.text((x, yy), seg, font=F_CODE, fill=TOK_COL[kind])
                x += len(seg) * CHAR_W
    d.rectangle((0, 0, PW, 52), fill=(14, 16, 26))           # title bar over the code
    for i, c in enumerate([(255, 95, 87), (254, 188, 46), (40, 200, 64)]):
        d.ellipse((20 + i * 26, 16, 36 + i * 26, 32), fill=c)
    d.text((110, 11), "meteors.py", font=F_UI, fill=(206, 208, 222))
    d.ellipse((PW - 250, 18, PW - 236, 32), fill=(40, 200, 64))
    d.text((PW - 228, 11), "running  30 fps", font=F_UI, fill=(146, 150, 170))
    d.line((0, 46, PW, 46), fill=(34, 38, 58), width=2)
    mask = Image.new("L", (PW, PH_), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, PW - 1, PH_ - 1), 18, fill=255)
    img.paste(panel, (PX0, PY0), mask)
    ImageDraw.Draw(img).rounded_rectangle((PX0, PY0, PX0 + PW - 1, PY0 + PH_ - 1), 18,
                                         outline=(38, 44, 66), width=2)


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
        encode(sys.argv[1] if len(sys.argv) > 1 else "meteors.mp4")
