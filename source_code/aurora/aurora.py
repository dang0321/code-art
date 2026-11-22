"""aurora.py - the northern lights over a mountain lake, drawn with ASCII
characters and rendered straight to a 1080x1920 video for Reels.

    python aurora.py                  -> aurora.mp4
    python aurora.py out.mp4          -> custom output name
    python aurora.py --still 10 a.png -> one frame at t=10s
"""
import math, os, re, shutil, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops

W, H, FPS, DUR = 1080, 1920, 30, 20.0
TAU = math.tau
rng = np.random.default_rng(2026)

CW, CH = 10, 16
COLS, ROWS = W // CW, 75
GY0 = 140
H0 = 50                                   # far shore of the lake
WATER_ROWS = ROWS - H0
BG = (3, 6, 14)

GREEN = np.array([0.32, 1.0, 0.55])       # oxygen 557.7 nm
TOP = np.array([0.82, 0.28, 0.72])        # oxygen 630.0 nm + sunlit nitrogen
PINK = np.array([1.0, 0.36, 0.62])        # nitrogen fringe at the lower edge
AUR = np.array(list(".,'`:;!i|l"))


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


def key(frames, t):                       # eased keyframes [(time, value), ...]
    if t <= frames[0][0]:
        return frames[0][1]
    for (t0, v0), (t1, v1) in zip(frames, frames[1:]):
        if t < t1:
            return v0 + (v1 - v0) * float(smooth((t - t0) / (t1 - t0)))
    return frames[-1][1]


# --- sky: a few hundred twinkling stars
N_STAR = 340
ST_C, ST_R = rng.integers(0, COLS, N_STAR), rng.integers(0, H0 - 4, N_STAR)
ST_CH = rng.choice(list(".`'.+.*"), N_STAR)
ST_PH, ST_B = rng.uniform(0, TAU, N_STAR), rng.uniform(0.25, 1.0, N_STAR) ** 2

# --- land: snowy peaks on the far shore, a pine line and a little cabin
PEAKS = [(6, 5, 12), (24, 8, 16), (44, 5, 10), (64, 10, 18), (82, 7, 13), (101, 5, 11)]
cc = np.arange(COLS)
MH = np.max([h * (1 - np.abs(cc - c) / w) for c, h, w in PEAKS], axis=0)
MH = np.clip(np.round(MH + np.sin(cc * 0.9) * 0.6), 2, None).astype(int)
LAND = {}                                 # (row, col) -> (char, kind) 0 edge/1 snow/2 rock/3 tree/4 wall/5 window
for c in range(COLS):
    top = H0 - MH[c]
    lft, rgt = MH[c - 1] if c else MH[c], MH[c + 1] if c + 1 < COLS else MH[c]
    edge ="/" if rgt > MH[c] else "\\" if lft > MH[c] else "^" if lft < MH[c] > rgt else "_"
    for r in range(top, H0):
        depth = r - top
        if depth == 0:
            LAND[r, c] = (edge, 0)
        elif depth < max(1, MH[c] * 0.45) and MH[c] > 4:
            LAND[r, c] = (":." [int(hash01(r, c) * 2)] if hash01(c, r) > 0.3 else "'", 1)
        else:
            LAND[r, c] = ("." if hash01(r, c, 1) > 0.75 else " ", 2)
for c in range(COLS):                     # pines along the shore
    if hash01(c, 7) > 0.45:
        tall = hash01(c, 9) > 0.6
        LAND[H0 - 1, c] = ("A" if tall else "^", 3)
        if tall:
            LAND[H0 - 2, c] = ("^", 3)
CABIN = [(" /\\ ", 4), ("/__\\", 4), ("|[]|", 4)]
for i, (row, _) in enumerate(CABIN):
    for j, ch in enumerate(row):
        if ch != " ":
            LAND[H0 - 3 + i, 15 + j] = (ch, 5 if ch in "[]" else 4)
LAND_RC = np.array(list(LAND.keys()))
LAND_CH = np.array([v[0] for v in LAND.values()])
LAND_K = np.array([v[1] for v in LAND.values()])

# --- lake: still water mirrors the sky, angle for angle
WR_C, WR_R = rng.integers(0, COLS, 240), rng.integers(H0, ROWS, 240)
WR_PH = rng.uniform(0, TAU, 240)

# --- camera: a pinhole on the shore, tilted up; the far shore sits on row H0
F, PITCH, EARTH_R = 700.0, 0.55, 6371.0
CP, SP = math.cos(PITCH), math.sin(PITCH)
CX, CY = W / 2, GY0 + H0 * CH - F * math.tan(PITCH)


def project(X, Y, Z):
    Y = Y - (X * X + Z * Z) / (2 * EARTH_R)   # the Earth curves away below the aurora
    yc, zc = Y * CP - Z * SP, Y * SP + Z * CP
    return CX + F * X / zc, CY - F * yc / zc, zc


def mirror_row(r):
    e = PITCH + math.atan((CY - (GY0 + (r + 0.5) * CH)) / F)
    return int(round((CY - F * math.tan(-e - PITCH) - GY0) / CH - 0.5))


RS = np.array([mirror_row(r) for r in range(H0, ROWS)])

# --- curtain: a ribbon of light 100 to 400 km up, sampled as a sheet of points
NS, NA = 1600, 80


def ribbon(seed, L, z, act, fold, drift, tilt=0.0, bend=0.2, x0=0.0):
    r = np.random.default_rng(seed)
    s = np.linspace(-1, 1, NS) + r.uniform(-1, 1, NS) / NS
    a = np.linspace(0, 330, NA)[None, :] + r.uniform(-2, 2, (NS, NA))
    return dict(seed=seed, L=L, z=z, act=act, fold=fold, drift=drift, tilt=tilt, bend=bend, x0=x0, s=s, a=a,
                ph=r.uniform(0, TAU, 4))


def curtain(R, t):
    act = key(R["act"], t)
    if act < 0.01:
        return None
    s, A, ph = R["s"], R["a"], R["ph"]
    m = float(smooth((act - 0.8) / 1.0)) * 0.85   # how stormy this band is

    # --- fold: the sheet curls over itself and the curls roll along it
    C = key(R["fold"], t)
    wave = 3.6 * s + 0.7 * t + ph[0]
    X = R["x0"] + s * R["L"] + C * np.sin(wave) + R["drift"] * t
    Z0 = key(R["z"], t)                     # the band runs at a slant, bending like a slow S
    Z = Z0 * (1 + R["tilt"] * s + R["bend"] * np.sin(2.4 * s + 0.25 * t + ph[1])) + 0.6 * C * np.cos(wave)
    hb = 105 + 5 * np.sin(9 * s + 1.1 * t + ph[2])

    # --- rays: thin field-aligned streaks that race along the curtain and flicker
    q = s * 55 + t * (1.5 + 4 * m) + ph[3] * 10
    fl = np.floor(q)
    hr = hash01(fl, R["seed"])
    flick = 0.65 + 0.35 * np.sin(t * (4 + 6 * hr) + hr * 40)
    ray = (1 - m) + m * (0.15 + 1.7 * hr * flick * np.exp(-((q - fl - 0.5) / 0.24) ** 2))
    hs = 38 * (0.6 + 0.9 * hash01(fl, R["seed"] + 3)) * (0.8 + 0.4 * m)
    env = smooth((1 - np.abs(s)) / 0.3) * (0.6 + 0.4 * np.sin(1.7 * s + 0.3 * t + ph[1]))
    env = env * (1 + 0.9 * m * np.maximum(0, np.sin(5 * s - 3.2 * t + ph[2])) ** 6)

    # --- pulsate: late in the night the band breaks into patches that blink
    pm = float(smooth((t - 15.5) / 2.0))
    if pm > 0:
        hp = hash01(np.floor(s * 16 + ph[0]), R["seed"] + 11)
        blink = 0.5 + 0.5 * np.sin(TAU * t / (2.2 + 2 * hp) + hp * TAU)
        env = env * ((1 - pm) + pm * (0.1 + 1.2 * blink ** 3))

    # --- color: green from 100 km, red-violet tops, a pink hem when it erupts
    green = smooth(A / 6) * np.exp(-A / hs[:, None])
    top = 0.07 * (1 + 2.5 * m) * np.exp(-((A - 150) / 65) ** 2)
    pink = max(act - 1.4, 0) * 0.6 * np.exp(-((A - 1) / 4) ** 2)
    sx, sy, zc = project(X[:, None], hb[:, None] + A, Z[:, None])
    w = (act * env * ray)[:, None] * 1e5 / zc ** 2   # surface brightness, not point count
    return sx, sy, w * green, w * top, w * pink


# --- breakup: the storm, stacked arcs that erupt at t = 7.5 s
RIBBONS = [
    # main arc and its twin just behind it: a double arc from the first frame
    ribbon(1, 750, z=[(0, 360), (6, 320), (10, 280), (14, 235), (20, 260)], tilt=0.45, bend=0.38,
           act=[(0, 1.4), (5, 1.4), (7.0, 1.3), (8.0, 2.2), (11, 2.0), (14, 1.6), (16.5, 1.3), (20, 1.1)],
           fold=[(0, 20), (6, 70), (8.5, 230), (14, 260), (18, 150)], drift=-12),
    ribbon(4, 800, z=[(0, 480), (6, 440), (10, 400), (14, 340), (20, 360)], tilt=0.4, bend=0.3, x0=80,
           act=[(0, 1.0), (6, 1.1), (8.2, 1.9), (12, 1.7), (16, 1.2), (20, 0.9)],
           fold=[(0, 30), (8.5, 180), (14, 220), (20, 140)], drift=-8),
    # a band high on the left, and one sweeping in from the far right
    ribbon(5, 450, z=[(0, 260), (8, 230), (14, 200), (20, 220)], tilt=-0.3, bend=0.3, x0=-250,
           act=[(0, 0.6), (4, 0.9), (8.3, 1.8), (13, 1.6), (17, 1.0), (20, 0.8)],
           fold=[(0, 40), (9, 160), (20, 120)], drift=6),
    ribbon(6, 500, z=[(0, 700), (6, 500), (10, 300), (16, 250), (20, 280)], tilt=0.3, bend=0.35, x0=300,
           act=[(0, 0), (3, 0), (5.5, 1.0), (9, 1.8), (14, 1.4), (20, 0.9)],
           fold=[(0, 60), (10, 200), (20, 140)], drift=-15),
    # a long low glow along the horizon behind the peaks
    ribbon(2, 1500, z=[(0, 900), (20, 720)], tilt=-0.15, bend=0.12,
           act=[(0, 0.7), (7, 0.9), (9.5, 1.3), (15, 1.0), (20, 0.8)],
           fold=[(0, 80), (10, 200), (20, 120)], drift=10),
]
# --- corona: two more bands race in from both sides and pass almost overhead
RIBBONS += [
    ribbon(3, 650, z=[(7.5, 560), (10.5, 360), (13.5, 190), (16, 150), (20, 140)], tilt=-0.4, bend=0.25,
           act=[(0, 0), (7.6, 0), (8.6, 1.2), (12, 1.5), (15, 1.3), (17.5, 0.8), (20, 0.5)],
           fold=[(0, 90), (12, 150), (20, 110)], drift=18),
    ribbon(7, 600, z=[(10, 450), (13, 250), (16, 170), (20, 160)], tilt=0.35, bend=0.3, x0=-150,
           act=[(0, 0), (10.3, 0), (11.5, 1.2), (14, 1.5), (17, 0.9), (20, 0.5)],
           fold=[(0, 80), (14, 150), (20, 100)], drift=-20),
]
KP = [(0, 3.1), (6, 4.0), (7.5, 5.6), (9, 7.0), (13, 7.7), (16, 6.0), (20, 4.4)]
GAIN = 0.45


def blur(a, axis, n=1):
    for _ in range(n):
        p = np.pad(a, [(1, 1) if ax == axis else (0, 0) for ax in range(2)], mode="edge")
        sl = [tuple(slice(i, i + a.shape[ax]) if ax == axis else slice(None) for ax in range(2))
              for i in range(3)]
        a = 0.25 * p[sl[0]] + 0.5 * p[sl[1]] + 0.25 * p[sl[2]]
    return a


def aurora(t):
    acc = np.zeros((3, ROWS * COLS))
    for R in RIBBONS:
        out = curtain(R, t)
        if out is None:
            continue
        sx, sy = out[0].ravel(), out[1].ravel()
        ci, ri = np.floor(sx / CW).astype(int), np.floor((sy - GY0) / CH).astype(int)
        ok = (ci >= 0) & (ci < COLS) & (ri >= 0) & (ri < H0)
        idx = (ri * COLS + ci)[ok]
        for k in range(3):
            acc[k] += np.bincount(idx, out[2 + k].ravel()[ok], ROWS * COLS)
    acc = acc.reshape(3, ROWS, COLS)
    acc = np.stack([blur(a, 0, 2) for a in acc])
    B = acc.sum(0)
    col = (acc[0, ..., None] * GREEN + acc[1, ..., None] * TOP + acc[2, ..., None] * PINK) \
        / np.maximum(B, 1e-9)[..., None]
    lum = 1 - np.exp(-GAIN * B ** 0.6)          # soft tone curve: keeps the bright rays apart
    lum *= 0.95 + 0.1 * hash01(np.arange(COLS)[None, :], np.arange(ROWS)[:, None], math.floor(t * 12))
    return lum, col, B


# --- meteor: one shooting star near the end
def meteor(t):
    s = (t - 16.9) / 0.6
    if not 0 <= s < 1.3:
        return []
    out = []
    for j in range(14):
        sj = s - j * 0.03
        if 0 <= sj <= 1:
            x, y = 780 - 330 * sj, 230 + 230 * sj
            v = (0.95 if j == 0 else 0.8 * math.exp(-j / 5)) * (1 - max(0, s - 1) / 0.3)
            out.append((x, y, "*" if j == 0 else "/", v))
    return out


def to_rgb(lum, col):
    c = col * (0.18 + 0.95 * lum)[..., None] + np.clip(lum - 0.88, 0, 1)[..., None] * 2.5
    return (np.clip(c, 0, 1) * 255).astype(int)


def compose(t):
    chars = np.full((ROWS, COLS), " ", "<U1")
    rgb = np.zeros((ROWS, COLS, 3), int)
    lum, col, B = aurora(t)

    # stars, washed out where the aurora is bright
    tw = 0.55 + 0.45 * np.sin(t * 2.3 + ST_PH)
    v = ST_B * tw * np.clip(1 - lum[ST_R, ST_C] * 2.5, 0, 1)
    sv = (40 + 190 * v).astype(int)
    on = v > 0.06
    chars[ST_R[on], ST_C[on]] = ST_CH[on]
    rgb[ST_R[on], ST_C[on]] = np.c_[sv, sv, sv + 20][on].clip(0, 255)

    sky = (lum > 0.1) & ((lum > 0.25) | (hash01(*np.indices(lum.shape), math.floor(t * 6)) < lum * 4.5))
    sky[H0:] = False
    chars[sky] = AUR[np.clip((lum[sky] * len(AUR)).astype(int), 0, len(AUR) - 1)]
    rgb[sky] = to_rgb(lum[sky], col[sky])

    for x, y, ch, v in meteor(t):
        c, r = int(x // CW), int((y - GY0) // CH)
        if 0 <= c < COLS and 0 <= r < H0:
            chars[r, c] = ch
            rgb[r, c] = (np.array([0.85, 0.92, 1.0]) * (0.2 + v) * 255).clip(0, 255).astype(int)

    # the landscape is lit by the aurora: green-tinted snow, dark pines, a warm window
    glow = float(np.clip(lum[:H0].mean() * 4, 0, 1))
    tint = (col[:H0] * lum[:H0, :, None]).sum((0, 1)) / max(lum[:H0].sum(), 1e-9)
    base = {0: (0.55, 0.6, 0.72), 1: (0.42, 0.47, 0.6), 2: (0.12, 0.14, 0.2),
            3: (0.07, 0.12, 0.11), 4: (0.3, 0.22, 0.17)}
    r, c = LAND_RC[:, 0], LAND_RC[:, 1]
    lc = np.array([base.get(k, (0, 0, 0)) for k in LAND_K]) * (0.55 + 0.6 * glow)
    lc += tint * glow * np.where(LAND_K <= 1, 0.35, 0.08)[:, None]
    win = LAND_K == 5
    lc[win] = np.array([1.0, 0.68, 0.32]) * (0.85 + 0.1 * math.sin(t * 3.1))
    chars[r, c] = LAND_CH
    rgb[r, c] = (lc.clip(0, 1) * 255).astype(int)

    # --- reflect: each water row shows the sky row at the mirrored angle, rippling
    rows = np.arange(H0, ROWS)
    depth = (rows - H0) / WATER_ROWS
    wob = np.round((0.3 + 1.3 * depth)[:, None] * np.sin(t * 1.6 + rows[:, None] * 1.3)
                   + 0.5 * np.sin(t * 2.4 + cc[None, :] * 0.35)).astype(int)
    src_c = (cc[None, :] - wob) % COLS
    src_r = np.broadcast_to(RS[:, None], src_c.shape)
    h = hash01(np.floor(cc[None, :] / 3 + t * 1.5 * np.sin(rows[:, None])), rows[:, None])
    keep = (chars[src_r, src_c] != " ") & (h > 0.12)
    dim = (0.5 - 0.22 * depth)[:, None]
    wc = np.where(h > 0.88, "~", np.where(h > 0.8, "-", chars[src_r, src_c]))
    wr = (rgb[src_r, src_c] * dim[..., None]).astype(int)
    sub = chars[H0:]
    sub[keep] = wc[keep]
    rgb[H0:][keep] = wr[keep]

    for cw, rw, ph in zip(WR_C, WR_R, WR_PH):      # faint ripples on empty water
        cw = int(cw + 3 * math.sin(t * 0.5 + ph)) % COLS
        if chars[rw, cw] == " " and math.sin(t * 1.3 + ph) > 0.2:
            chars[rw, cw] = "-" if ph > 2 else "~"
            rgb[rw, cw] = (np.array([22, 30, 50]) + tint * glow * 40).astype(int)
    return chars, rgb, lum


# code panel: this very file scrolls by, following what is happening on screen
SRC = open(__file__, encoding="utf-8").read().splitlines()
TOK = re.compile(r"(?P<com>#.*)|(?P<str>\"[^\"]*\"|'[^']*')"
                 r"|(?P<kw>\b(?:def|for|in|if|elif|else|return|import|from|as|while|"
                 r"with|lambda|not|and|or|None|True|False|class)\b)|(?P<num>\b\d+(?:\.\d+)?\b)")
TOK_COL = {"com": (106, 115, 135), "str": (152, 195, 121), "kw": (198, 120, 221),
           "num": (209, 154, 102), None: (171, 178, 191)}


def tokens(line):
    out, pos = [], 0
    for m in TOK.finditer(line):
        out += [(line[pos:m.start()], None), (m.group(), m.lastgroup)]
        pos = m.end()
    return out + [(line[pos:], None)]


SRC_TOK = [tokens(l) for l in SRC]
PHASES = [(0.0, "sky"), (0.4, "land"), (0.9, "lake"), (1.4, "camera"), (2.2, "curtain"),
          (4.4, "rays"), (6.6, "breakup"), (8.8, "fold"), (11.0, "corona"), (13.0, "color"),
          (15.4, "pulsate"), (16.8, "meteor"), (17.8, "reflect"), (19.2, "encode"), (DUR, "encode")]
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


PX, PY, PW, PH_ = 30, 1365, W - 60, 525
LH, CODE_TOP = 27, 52
CHAR_W = F_CODE.getlength("M")


def code_panel(t, img):
    panel = Image.new("RGB", (PW, PH_), (16, 18, 26))
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
            d.rectangle((0, yy - 2, PW, yy + LH - 3), fill=(30, 35, 52))
        d.text((12, yy), f"{li + 1:>3}", font=F_CODE, fill=(75, 82, 100))
        x = 70
        for seg, kind in SRC_TOK[li]:
            seg = seg[: max(0, int((PW - 20 - x) / CHAR_W))]
            if seg:
                d.text((x, yy), seg, font=F_CODE, fill=TOK_COL[kind])
                x += len(seg) * CHAR_W
    d.rectangle((0, 0, PW, 52), fill=(16, 18, 26))           # title bar over the code
    for i, c in enumerate([(255, 95, 87), (254, 188, 46), (40, 200, 64)]):
        d.ellipse((20 + i * 26, 16, 36 + i * 26, 32), fill=c)
    d.text((110, 11), "aurora.py", font=F_UI, fill=(200, 205, 215))
    d.ellipse((PW - 250, 18, PW - 236, 32), fill=(40, 200, 64))
    d.text((PW - 228, 11), "running  30 fps", font=F_UI, fill=(140, 150, 165))
    d.line((0, 46, PW, 46), fill=(36, 40, 54), width=2)
    mask = Image.new("L", (PW, PH_), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, PW - 1, PH_ - 1), 18, fill=255)
    img.paste(panel, (PX, PY), mask)
    ImageDraw.Draw(img).rounded_rectangle((PX, PY, PX + PW - 1, PY + PH_ - 1), 18,
                                         outline=(40, 46, 62), width=2)


# --- render: draw every lit cell, then bloom
def frame(t):
    chars, rgb, lum = compose(t)
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    for r, c in np.argwhere(chars != " ").tolist():
        d.text((c * CW, GY0 + r * CH), chars[r, c], font=F_GRID, fill=tuple(rgb[r, c].tolist()))

    box = (0, GY0 - 20, W, PY - 5)
    region = img.crop(box)
    base = ImageChops.subtract(region, Image.new("RGB", region.size, BG))
    glow = base.filter(ImageFilter.GaussianBlur(5)).point(lambda v: int(v * 0.8))
    halo = base.filter(ImageFilter.GaussianBlur(34)).point(lambda v: min(255, int(v * 2.2)))
    img.paste(ImageChops.add(ImageChops.add(region, glow), halo), box[:2])

    d = ImageDraw.Draw(img)
    d.text((36, 34), "aurora.py", font=F_UI, fill=(200, 205, 215))
    d.text((36, 66), f"Kp index  {key(KP, t):.1f}", font=F_UI, fill=(120, 128, 145))
    d.text((36, 98), f"lit cells {int((lum[:H0] > 0.05).sum()):,}", font=F_UI, fill=(120, 128, 145))
    ts = f"t {t:05.2f}s"
    d.text((W - 36 - F_UI.getlength(ts), 34), ts, font=F_UI, fill=(120, 128, 145))
    code_panel(t, img)
    return img


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
    else:
        encode(sys.argv[1] if len(sys.argv) > 1 else "aurora.mp4")
