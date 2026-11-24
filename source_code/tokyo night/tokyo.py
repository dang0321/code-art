"""tokyo.py - Tokyo across the bay at dusk. Mount Fuji goes dark against the afterglow, the city
switches on window by window, then the Rainbow Bridge, Tokyo Tower and the Skytree light up in
turn, and all of it is doubled, broken and swaying, in the water.
Drawn with ASCII characters and rendered straight to a 1080x1920 video for Reels.

    python tokyo.py                     -> tokyo.mp4
    python tokyo.py out.mp4             -> custom output name
    python tokyo.py --still 10 a.png    -> one frame at t=10s
    python tokyo.py --stills 2,9,13 dir -> several frames into a folder
"""
import math, os, re, shutil, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops

W, H, FPS, DUR = 1080, 1920, 30, 20.0
TAU = math.tau
CW, CH = 10, 16
COLS, ROWS = W // CW, 75
GY0, GH = 140, 75 * 16
CXs = np.arange(COLS) * CW + CW / 2                 # cell centres on screen
CYs = GY0 + np.arange(ROWS) * CH + CH / 2
CX, CY = np.meshgrid(CXs, CYs)


def font(names, size):
    for name in names:
        for d in ("C:/Windows/Fonts", "/usr/share/fonts/truetype/dejavu", "/Library/Fonts"):
            p = os.path.join(d, name)
            if os.path.exists(p):
                return ImageFont.truetype(p, size)
    return ImageFont.load_default(size)


MONO = ["consola.ttf", "DejaVuSansMono.ttf", "Menlo.ttc"]
F_GRID, F_UI, F_CODE = font(MONO, 16), font(MONO, 26), font(MONO, 21)
F_KANA = font(["YuGothM.ttc", "msgothic.ttc", "NotoSansCJK-Regular.ttc"], 18)   # half-width katakana: one cell each


def hash01(a, b, c=0.0):                  # cheap deterministic noise in [0, 1)
    v = np.sin(a * 12.9898 + b * 78.233 + c * 37.719) * 43758.5453
    return v - np.floor(v)


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def hsv(h):
    k = (np.array([0, 4, 2]) + np.asarray(h)[..., None] * 6) % 6
    return 1 - np.clip(np.minimum(k, 4 - k), 0, 1)


# the evening, in order
BRIDGE_ON, TOWER_ON, SKYTREE_ON, RAINBOW_ON = 4.0, 8.5, 12.0, 15.5

# --- sky: the sun goes down exactly behind Fuji's summit, then we watch the dusk deepen
SHORE = 940                                  # the far shore, where the city stands on the water
R_SHORE = int((SHORE - GY0) // CH)           # first row of water
PX_DEG = 27.0                                # screen pixels per degree of view
FUJI_X, FUJI_TOP, FUJI_H, FUJI_S = 235, 650, 290, 230
SUN_X = FUJI_X                               # a "Diamond Fuji": the sun sets right on the summit
DS = [1.2, 2.5, 4.5, 6.5, 8.0]               # afterglow colour as the sun sinks: gold, orange, red, purple, gone
GLOW = np.array([[1.0, 1.0, 0.95, 0.62, 0.3], [0.7, 0.52, 0.32, 0.17, 0.08], [0.3, 0.2, 0.2, 0.3, 0.2]])


def sun_dep(t):                              # degrees below the horizon (a time-lapse of half an hour)
    return -0.6 + 8.1 * t / DUR


def sky_field(t):
    e = np.maximum((SHORE - CY) / PX_DEG, 0.0)[..., None]
    a = ((CX - SUN_X) / PX_DEG)[..., None]
    d = max(sun_dep(t), 1.2)                 # the sky itself keeps its last light a while
    fade = math.exp(-(d - 2) / 3.2)
    zen = np.array([0.05, 0.09, 0.24]) * fade + np.array([0.006, 0.009, 0.03])
    hor = np.array([0.15, 0.19, 0.38]) * fade + np.array([0.01, 0.014, 0.035])
    k = np.exp(-e / 12)
    col = zen * (1 - k) + hor * k
    # the afterglow hugs the horizon above the sun, reddening and sinking as the sun goes deeper
    gc = np.array([np.interp(d, DS, GLOW[i]) for i in range(3)])
    g = 1.05 * math.exp(-(d - 1.2) / 2.8) * np.exp(-e / (7.5 - 0.55 * (d - 2))) * np.exp(-(a / (30 + 3 * d)) ** 2)
    col = col * (1 - 0.6 * np.minimum(g / 0.4, 1)) + g * gc         # where the glow is, the blue gives way
    # opposite the sun, the earth's own shadow rises up the sky under the pink Belt of Venus
    east = smooth((CX - 520) / 420)[..., None]
    top = 0.9 + 0.55 * d
    belt = 0.08 * math.exp(-(d - 2) / 2.2) * np.exp(-((e - top - 3.0) / 3.5) ** 2) * east
    col = col + belt * np.array([0.9, 0.5, 0.62])
    return col * (1 - 0.3 * east * (e < top))


SKY_RAMP = np.array(list("..`'--~~=="))


def sky_layer(S, t):
    v = sky_field(t)
    up = CY < SHORE
    lum = v.max(-1)
    hh = hash01(CX, CY, 3.0)
    draw = up & ((lum > 0.36) | (hh < lum * 1.2))       # the open sky is mostly wash, with a sprinkle of grain
    S.ch[draw] = SKY_RAMP[np.clip((lum[draw] * 14).astype(int), 0, 9)]
    S.val[up] = v[up] * 0.9
    S.wash[up] = v[up]


# --- stars: as the twilight drains away the first stars come out above the city's glow
STARS = dict(x=np.random.default_rng(9).uniform(0, W, 110), y=GY0 + 10 + np.random.default_rng(10).uniform(0, 520, 110) ** 1.15 / 1.3,
             m=np.random.default_rng(11).uniform(0.2, 1.0, 110) ** 2, ph=np.random.default_rng(12).uniform(0, TAU, 110))


def stars_layer(S, t):
    vis = smooth((sun_dep(t) - 5.0) / 3.0)
    if vis <= 0:
        return
    c = np.clip((STARS["x"] // CW).astype(int), 0, COLS - 1)
    r = np.clip(((STARS["y"] - GY0) // CH).astype(int), 0, ROWS - 1)
    b = vis * STARS["m"] * (0.7 + 0.3 * np.sin(2.3 * t + STARS["ph"])) * 0.5
    ok = (b > 0.03) & (CYs[r] < SHORE - 120)
    S.ch[r[ok], c[ok]] = np.where(b[ok] > 0.3, "+", ".")
    S.val[r[ok], c[ok]] = b[ok, None] * np.array([0.85, 0.9, 1.0])


# --- plane: an airliner on its way down into Haneda, landing light on, beacons blinking
def plane_layer(S, t):
    if not 1.0 < t < 19.5:
        return
    x, y = 60 + 50 * (t - 1), 330 + 7 * (t - 1)
    c, r = int(x // CW), int((y - GY0) // CH)
    if 0 <= c < COLS - 1:
        S.ch[r, c], S.val[r, c] = "*", np.array([1.0, 0.97, 0.9]) * 0.75
        if (t * 1.1) % 1 < 0.14:
            S.ch[r, c - 1], S.val[r, c - 1] = ".", np.array([1.0, 0.15, 0.1]) * 0.8
        if (t * 0.8) % 1 < 0.06 or 0.12 < (t * 0.8) % 1 < 0.17:
            S.ch[r, c + 1], S.val[r, c + 1] = "'", np.array([1.0, 1.0, 1.0])


# --- fuji: a steep summit cone flaring into long concave flanks, dark against the afterglow,
# its snowcap scored by ravines; the tower stands in front of its right flank
def ridge(x):
    dx = np.abs(x - FUJI_X)
    return FUJI_TOP + FUJI_H * (1 - np.exp(-np.maximum(dx - 18, 0) / FUJI_S))


def snowline(x):
    return FUJI_TOP + 72 + 24 * np.abs(np.sin((x - FUJI_X) * 0.11)) - 0.12 * np.abs(x - FUJI_X)


_sub = CY[..., None] + np.array([-6.0, -2.0, 2.0, 6.0])
FUJI_COV = (_sub > ridge(CX)[..., None]).mean(-1) * (CY < SHORE)
_slope = (ridge(CX + 5) - ridge(CX - 5)) / 10


def fuji_layer(S, t):
    d = sun_dep(t)
    rim = (FUJI_COV > 0) & (FUJI_COV < 0.75)
    body = FUJI_COV >= 0.75
    S.ch[body] = np.where(hash01(CX[body], CY[body]) < 0.5, ":", ".")
    S.val[body] = np.array([0.04, 0.045, 0.075]) * (0.5 + 0.5 * math.exp(-(d - 2) / 4))
    S.wash[body | rim] *= 0.15
    # the snowcap, scored by ravines running down from the crater, catches what light the sky has left
    cap = body & (CY < snowline(CX))
    gully = np.sin((CX - FUJI_X) * 0.36 + (CY - FUJI_TOP) * 0.05) > -0.1
    S.ch[cap] = np.where(gully[cap], np.where(CX[cap] < FUJI_X - 8, "/", np.where(CX[cap] > FUJI_X + 8, "\\", "|")), "'")
    sky_light = 0.1 + 0.16 * math.exp(-max(d - 1.2, 0) / 3.5)
    S.val[cap] = np.array([0.72, 0.78, 1.0]) * sky_light * np.where(gully[cap], 1.0, 0.55)[:, None]
    # the ridge line is lit from behind by the afterglow
    sl = _slope[rim]
    S.ch[rim] = np.where(sl < -0.3, "/", np.where(sl > 0.3, "\\", "_"))
    S.val[rim] = sky_field(t)[rim] * 1.35 + 0.03
    # Diamond Fuji: the last of the sun sits on the summit like a jewel, flares, and is gone
    I = 3.2 * (1 - smooth(t / 2.2))
    if I > 0.02:
        gold = np.array([1.0, 0.86, 0.55])
        r0, c0 = int((FUJI_TOP - 4 - GY0) // CH), int(FUJI_X // CW)
        S.ch[r0, c0], S.val[r0, c0] = "*", gold * I
        for k in range(1, 8):
            a = I * 0.6 / k ** 0.9
            for dc in (-k, k):
                S.ch[r0, c0 + dc], S.val[r0, c0 + dc] = "-", gold * a
            if k <= 3:
                for dr, dc, ch in ((-k, 0, "|"), (-k, -k, "\\"), (-k, k, "/")):
                    S.ch[r0 + dr, c0 + dc], S.val[r0 + dr, c0 + dc] = ch, gold * a * 0.8
        near = (CY < SHORE) & ~(FUJI_COV > 0.5)
        halo = I * 0.3 * np.exp(-np.hypot(CX - FUJI_X, (CY - FUJI_TOP) * 1.3) / 150)
        S.wash[near] += gold * halo[near][:, None]


# --- city: three depths of buildings on the far shore, and Shiba Park's trees round the tower
WIN_PAL = np.array([[1.0, 0.86, 0.6], [0.86, 0.95, 1.0], [1.0, 0.72, 0.38], [0.7, 0.88, 1.0], [1.0, 0.95, 0.82]])
TT_X, TT_B, TT_H = 465, SHORE - 4, 560            # Tokyo Tower: 333 m, 1.68 px to the metre
ST_X, ST_B, ST_H = 875, SHORE - 4, 700            # Tokyo Skytree: 634 m, the tallest tower in the world


def make_city():
    r = np.random.default_rng(3)
    C = dict(ch=np.full((ROWS, COLS), "", "<U1"), base=np.zeros((ROWS, COLS, 3)), on=np.full((ROWS, COLS), np.inf),
             wcol=np.zeros((ROWS, COLS, 3)), wch=np.full((ROWS, COLS), "", "<U1"), mask=np.zeros((ROWS, COLS), bool))
    roofs = []

    def building(x0, x1, top, layer):
        cs = np.where((CXs >= x0) & (CXs < x1))[0]
        rs = np.where((CYs >= top) & (CYs < SHORE))[0]
        if len(cs) < 2 or len(rs) < 1:
            return
        dim = (0.45, 0.72, 1.0)[layer]
        tex, style = r.choice([":", ".", "'", ","]), r.random()
        for rr in rs:
            for cc in cs:
                roof, side = rr == rs[0], cc in (cs[0], cs[-1])
                C["ch"][rr, cc] = "_" if roof else "|" if side else tex
                C["base"][rr, cc] = np.array([0.3, 0.36, 0.58]) * (0.075 if roof else 0.055 if side else 0.035) * dim
                C["on"][rr, cc], C["wch"][rr, cc] = np.inf, ""
                C["mask"][rr, cc] = True
                if not roof and not side and r.random() < (0.62 if layer == 0 else 0.86):
                    C["on"][rr, cc] = r.uniform(-4, 0) if r.random() < 0.22 else 0.3 + 13.5 * r.random() ** 1.3
                    C["wcol"][rr, cc] = WIN_PAL[r.integers(len(WIN_PAL))] * r.uniform(0.3, 0.75) * dim
                    C["wch"][rr, cc] = "=" if style < 0.45 else "#" if style < 0.7 else ":"
        if SHORE - top > 140:
            roofs.append((cs[len(cs) // 2], rs[0] - 1))

    for layer, (wmin, wmax, hmin, hmax, step) in enumerate(((16, 46, 16, 70, 0.8), (24, 66, 28, 105, 1.2), (36, 80, 60, 150, 2.6))):
        x = -20 + r.uniform(0, 40)
        while x < W + 20:
            w = r.uniform(wmin, wmax)
            mid = x + w / 2
            cluster = (1 + 2.3 * math.exp(-((mid - 680) / 120) ** 2)                 # Shiodome and Shinagawa
                       + 0.8 * math.exp(-((mid - 1000) / 90) ** 2))
            park = 1 - 0.8 * math.exp(-((mid - TT_X) / 80) ** 2)                      # Shiba Park keeps it low
            tree = 1 - 0.6 * math.exp(-((mid - ST_X) / 40) ** 2)
            view = 1 - 0.6 * math.exp(-((mid - FUJI_X) / 200) ** 2)                   # keep the view of Fuji open
            h = r.uniform(hmin, hmax) * cluster * park * tree * view
            if layer < 2 or (r.random() < 0.35 * cluster and abs(mid - FUJI_X) > 260):
                building(x, x + w, SHORE - h, layer)
            x += w * r.uniform(0.7, 1.0) * step
    # the trees of Shiba Park round the foot of the tower
    tops = SHORE - 30 - 9 * np.sin(CX * 0.07) - 6 * np.sin(CX * 0.19 + 1)
    park = (np.abs(CX - TT_X) < 160) & (CY > tops) & (CY < SHORE)
    C["ch"][park] = np.where(hash01(CX[park], CY[park], 5) < 0.5, "&", "%")
    C["base"][park] = np.array([0.05, 0.1, 0.065]) * 0.5
    C["on"][park] = np.inf
    C["mask"] |= park
    C["park"] = park
    C["roofs"] = roofs
    return C


CITY = make_city()


def blink(t):                                # Tokyo's red obstruction lights breathe together
    return (0.5 + 0.5 * math.cos(TAU * 0.75 * t)) ** 3


def city_layer(S, t, tower_lit):
    C, m = CITY, CITY["mask"]
    lvl = smooth((t - C["on"]) / 0.12)
    lit = m & (lvl > 0)
    S.ch[m] = C["ch"][m]
    S.val[m] = C["base"][m]
    S.wash[m] *= 0.1
    S.ch[lit] = C["wch"][lit]
    S.val[lit] = C["wcol"][lit] * lvl[lit][:, None] + C["base"][lit]
    p = C["park"]                                 # the park glows orange once the tower is lit
    S.val[p] += tower_lit * np.array([1.0, 0.5, 0.15]) * 0.14 * np.exp(-np.abs(CX[p] - TT_X) / 90)[:, None]
    b = blink(t)
    for c, r in C["roofs"]:
        if r >= 0 and b > 0.05:
            S.ch[r, c], S.val[r, c] = "*" if b > 0.5 else ".", np.array([1.0, 0.12, 0.08]) * 0.8 * b
    return int((lit & (C["wch"] != "")).sum())


# --- neon: vertical neon signs down the faces of the nearer buildings, spelt in half-width
# katakana (one character a cell); each tube stutters as it strikes, then holds
SIGNS = [  # x, text, colour, the moment it is switched on
    (575, "ﾗｰﾒﾝ", (1.0, 0.2, 0.28), 3.2),      # ramen
    (655, "ｶﾗｵｹ", (1.0, 0.3, 0.85), 5.6),      # karaoke
    (745, "ﾎﾃﾙ", (0.35, 0.92, 1.0), 7.2),           # hotel
    (1015, "ｻｳﾅ", (1.0, 0.72, 0.2), 10.4),          # sauna
]


def place_signs():
    out = []
    for x, text, col, t_on in SIGNS:
        c = int(x // CW)
        rows = np.where(CITY["mask"][:, c] & ~CITY["park"][:, c])[0]
        if len(rows) >= len(text) + 3:
            out.append((c, rows[0] + 2, text, np.array(col), t_on))
    return out


SIGN_CELLS = place_signs()


def neon_layer(S, t):
    for c, r0, text, col, t_on in SIGN_CELLS:
        a = t - t_on
        if a < 0:
            continue
        lvl = 0.8 * float(hash01(c, math.floor(t * 18)) > 0.45) if a < 0.7 else 0.92 + 0.08 * math.sin(37 * t + c)
        for k, ch in enumerate(text):
            S.ch[r0 + k, c], S.val[r0 + k, c] = ch, col * 0.95 * lvl


# --- tower: Tokyo Tower, lattice steel on four splayed legs (80 m apart), a broad two-storey main
# deck at 150 m, a small top deck at 250 m and a banded mast; its Landmark Light climbs from the feet
# to the tip in warm sodium orange
def tt_w(z):                                  # half-width in pixels, from the real tower's dimensions
    return np.select([z < 0.435, z < 0.48, z < 0.745, z < 0.772, z < 0.84, z <= 1.0],
                     [21 + 46 * np.clip(1 - z / 0.435, 0, 1) ** 1.7, 33.0, 19 - 9 * (z - 0.48) / 0.265, 13.0,
                      8 - 4 * (z - 0.772) / 0.068, 1.5], 0.0)


def tt_arch(z):                               # the great arch between the legs
    return np.where(z < 0.16, 44 * np.sqrt(np.clip(1 - (z / 0.16) ** 2, 0, 1)), 0.0)


def tower_cells(X0, B0, H0, wfun, arch=None):
    z = (B0 - CY) / H0
    dx = CX - X0
    w = wfun(np.clip(z, 0, 1))
    inside = (z >= 0) & (z <= 1) & (np.abs(dx) <= w + 4)
    if arch is not None:
        inside &= ~(np.abs(dx) < arch(z) - 4)
    edge_l = inside & ~(np.abs(dx - CW) <= w + 4)        # nothing of the tower in the cell to the left
    edge_r = inside & ~(np.abs(dx + CW) <= w + 4)
    return inside, edge_l, edge_r, z


TT_IN, TT_EL, TT_ER, TT_Z = tower_cells(TT_X, TT_B, TT_H, tt_w, tt_arch)
LANDMARK = np.array([1.0, 0.44, 0.1])            # sodium lamps under the legs, shining up the steel
TT_DECK = lambda z: ((z >= 0.435) & (z < 0.48)) | ((z >= 0.745) & (z < 0.772))


def tt_chars():
    ch = np.full((ROWS, COLS), "", "<U1")
    z, rr, cc = TT_Z, np.arange(ROWS)[:, None] + 0 * CX, np.arange(COLS)[None, :] + 0 * CY
    ch[:] = np.where(rr % 3 == 0, "=", np.where((rr + cc) % 2 == 0, "X", "x"))
    ch = np.where(TT_EL, np.where(z < 0.435, "/", "|"), ch)          # the legs splay; above the deck it rises straight
    ch = np.where(TT_ER, np.where(z < 0.435, "\\", "|"), ch)
    deck = TT_DECK(z)
    ch = np.where(deck, np.where(TT_EL, "[", np.where(TT_ER, "]", "#")), ch)
    ch = np.where(z >= 0.84, "|", ch)
    return np.where(TT_IN, ch, "")


TT_CH = tt_chars()


def tower_layer(S, t):
    m, z = TT_IN, TT_Z
    f = (t - TOWER_ON) / 1.4
    lit = smooth((f - z[m]) / 0.06 + 0.5) if f > -0.1 else np.zeros(m.sum())
    over = 1 + 0.3 * math.exp(-max(t - TOWER_ON - 1.4, 0) / 0.7) * (t > TOWER_ON)
    deck = TT_DECK(z[m])
    edge = (TT_EL | TT_ER)[m]
    band = (z[m] >= 0.772) & ((np.arange(ROWS)[:, None] + 0 * CX)[m] % 2 == 0)   # the mast's white bands
    on = np.where(deck[:, None], np.array([1.0, 0.92, 0.74]) * 1.05,
                  np.where(band[:, None], np.array([1.0, 0.9, 0.8]) * 0.8,
                           LANDMARK * (1.2 - 0.55 * z[m])[:, None] * np.where(edge, 1.25, 1.0)[:, None]))
    off = np.array([0.32, 0.13, 0.08]) * 0.11 * (0.4 + 0.6 * math.exp(-(sun_dep(t) - 2) / 4))
    S.ch[m] = TT_CH[m]
    S.val[m] = off * (1 - lit[:, None]) + on * (lit * over)[:, None] * 0.95
    S.wash[m] *= 0.1
    b = blink(t)                                   # obstruction lights on the decks and the mast
    for zz, dxs in ((0.995, (0,)), (0.775, (-1, 1)), (0.485, (-3, 3))):
        r, c0 = int((TT_B - zz * TT_H - GY0) // CH), TT_X // CW
        for dc in dxs:
            if b > 0.05:
                S.ch[r, c0 + dc], S.val[r, c0 + dc] = "*", np.array([1.0, 0.12, 0.08]) * 0.9 * b
    return float(smooth(f))


# --- skytree: Tokyo Skytree, a 68 m triangle at its foot rounding into a slim shaft, the wide
# Tembo Deck at 350 m, the Galleria ring at 450 m and the gain tower; its "Iki" light is the pale
# blue of the Sumida river, and it climbs the shaft too
def st_w(z):                                  # half-width in pixels, from the real tower's dimensions
    return np.select([z < 0.54, z < 0.575, z < 0.70, z < 0.722, z < 0.80, z <= 1.0],
                     [15 + 22 * np.clip(1 - z / 0.54, 0, 1) ** 1.3, 27.0, 13.0, 18.0, 8 - 2 * (z - 0.722) / 0.078, 2.0], 0.0)


ST_IN, ST_EL, ST_ER, ST_Z = tower_cells(ST_X, ST_B, ST_H, st_w)
IKI = np.array([0.5, 0.82, 1.0])
ST_DECK = lambda z: ((z >= 0.54) & (z < 0.575)) | ((z >= 0.70) & (z < 0.722))


def st_chars():
    z, rr = ST_Z, np.arange(ROWS)[:, None] + 0 * CX
    ch = np.where(rr % 2 == 0, "X", "|")
    ch = np.where(ST_EL, np.where(z < 0.54, "/", "|"), ch)
    ch = np.where(ST_ER, np.where(z < 0.54, "\\", "|"), ch)
    ch = np.where(ST_DECK(z), np.where(ST_EL, "(", np.where(ST_ER, ")", "#")), ch)     # the round decks
    ch = np.where(z >= 0.80, "|", ch)
    return np.where(ST_IN, ch, "")


ST_CH = st_chars()


def skytree_layer(S, t):
    m, z = ST_IN, ST_Z
    f = (t - SKYTREE_ON) / 1.6
    lit = smooth((f - z[m]) / 0.06 + 0.5) if f > -0.1 else np.zeros(m.sum())
    deck = ST_DECK(z[m])
    on = np.where(deck[:, None], np.array([0.92, 0.96, 1.0]), IKI * (1.05 - 0.3 * z[m])[:, None]) * 0.88  # a little hazed by distance
    off = np.array([0.15, 0.18, 0.26]) * 0.1 * (0.4 + 0.6 * math.exp(-(sun_dep(t) - 2) / 4))
    S.ch[m] = ST_CH[m]
    S.val[m] = off * (1 - lit[:, None]) + on * lit[:, None]
    S.wash[m] *= 0.1
    b = blink(t + 0.4)
    for zz in (0.998, 0.80, 0.722, 0.575):
        r = int((ST_B - zz * ST_H - GY0) // CH)
        if b > 0.05:
            S.ch[r, ST_X // CW], S.val[r, ST_X // CW] = "*", np.array([1.0, 0.12, 0.08]) * 0.85 * b
    return float(smooth(f))


# --- bay: Tokyo Bay mirrors the shore; the near water is rougher, so the copy breaks up and sways
def bay_layer(S, t):
    src = S.val[:R_SHORE].copy()
    lum_src = src.max(-1)
    cc = np.arange(COLS)
    for r in range(R_SHORE, ROWS):
        depth = (r - R_SHORE) / (ROWS - R_SHORE)
        rs = 2 * R_SHORE - 1 - r
        if rs < 1:
            break
        amp = 0.5 + 1.8 * depth
        off = np.round(amp * np.sin(1.3 * t + 0.55 * r + 0.08 * cc) + 0.6 * np.sin(2.1 * t - 0.31 * r + 0.23 * cc)).astype(int)
        c2 = np.clip(cc + off, 0, COLS - 1)
        rows = [max(rs - 1, 0), rs, min(rs + 1, R_SHORE - 1)]
        pick = np.argmax(np.stack([lum_src[q, c2] * w for q, w in zip(rows, (0.7, 1.0, 0.7))]), 0)
        v = np.stack([src[q, c2] for q in rows])[pick, cc]
        brk = 0.5 + 0.5 * np.sin(2.4 * t + 1.7 * r + 0.9 * cc + 6 * hash01(cc, r))
        v = v * (0.8 - 0.38 * depth) * (0.45 + 0.55 * brk)[:, None]
        lum = v.max(-1)
        hh = hash01(cc, r, math.floor(t * 6))
        S.val[r] = v
        S.ch[r] = np.where(lum > 0.2, np.where(hh < 0.5, "~", "="), np.where(lum > 0.06, np.where(hh < 0.5, "-", "~"),
                           np.where((lum > 0.015) | (hh < lum * 30), np.where(hh < 0.5, "-", "."), " ")))
        S.wash[r] = S.wash[rs] * 0.35


# --- bridge: the Rainbow Bridge, two decks hung from two towers; its lights sweep on along the
# cables, and late in the evening it lives up to its name
BR_TX, BR_TOP, BR_SAG, BR_DECK, BR_LOW, BR_WATER = (285, 795), 878, 1016, 1030, 1046, 1072


def cable_y(x):
    a, b = BR_TX
    mid, half = (a + b) / 2, (b - a) / 2
    return np.where(x < a, BR_TOP + (BR_DECK - BR_TOP) * np.clip((a - x) / (a + 20), 0, 1) ** 1.35,
                    np.where(x > b, BR_TOP + (BR_DECK - BR_TOP) * np.clip((x - b) / (W + 20 - b), 0, 1) ** 1.35,
                             BR_SAG - (BR_SAG - BR_TOP) * ((x - mid) / half) ** 2))


def make_bridge():
    ch = np.full((ROWS, COLS), "", "<U1")
    kind = np.zeros((ROWS, COLS), int)          # 1 tower, 2 cable, 3 hanger, 4 upper deck, 5 lower deck, 6 pier
    row = lambda y: int((y - GY0) // CH)
    for c in range(COLS):                         # the main and side cables, steep near the towers
        x = CXs[c]
        y0, y1 = cable_y(np.array(x - 5.0)), cable_y(np.array(x + 5.0))
        sl = (y1 - y0) / 10
        for r in range(row(min(y0, y1)), row(max(y0, y1)) + 1):
            ch[r, c] = "\\" if sl > 0.3 else "/" if sl < -0.3 else ("-" if ((cable_y(np.array(x)) - GY0) % CH) < 8 else "_")
            kind[r, c] = 2
        if BR_TX[0] + 25 < x < BR_TX[1] - 25 and c % 3 == 0:       # hangers down to the deck
            for r in range(row(cable_y(np.array(x))) + 1, row(BR_DECK)):
                ch[r, c], kind[r, c] = "|", 3
    rd, rl, rw = row(BR_DECK), row(BR_LOW), row(BR_WATER)
    ch[rd, :], kind[rd, :] = "=", 4
    ch[rl, :], kind[rl, :] = "-", 5
    for tx in BR_TX:                              # each tower: two legs, cross-beams, a pier into the bay
        for c in (tx // CW - 1, tx // CW + 1):
            for r in range(row(BR_TOP), rw + 1):
                ch[r, c], kind[r, c] = ("|", 1) if r < rw else ("#", 6)
        for y in (BR_TOP, 905, 975):
            ch[row(y), tx // CW], kind[row(y), tx // CW] = "=", 1
        ch[rw, tx // CW], kind[rw, tx // CW] = "#", 6
    return ch, kind


BR_CH, BR_KIND = make_bridge()
CARS = dict(x0=np.random.default_rng(21).uniform(0, W + 200, 16), v=np.random.default_rng(22).uniform(55, 95, 16) * np.where(np.arange(16) % 2, 1, -1))


def bridge_layer(S, t):
    m = BR_KIND > 0
    sweep = smooth((CX - ((t - BRIDGE_ON) / 1.6 * 1200 - 60)) / -60)      # the lights come on left to right
    white = np.array([0.92, 0.95, 1.0])
    rb = smooth((t - RAINBOW_ON) / 1.8)
    tint = (1 - 0.5 * rb)[..., None] * white + (0.5 * rb)[..., None] * hsv((CX / W * 0.85 + 0.05 * t) % 1)   # pastel
    lvl = np.select([BR_KIND == 1, BR_KIND == 2, BR_KIND == 3, BR_KIND == 4, BR_KIND == 5, BR_KIND == 6],
                    [0.75, 0.85, 0.1, 0.3, 0.1, 0.25], 0.0)
    dark = np.array([0.18, 0.2, 0.26]) * 0.12
    val = dark + (tint * (lvl * sweep)[..., None])
    deck = BR_KIND == 4                            # sodium street lamps along the road deck
    lamp = (0.18 + 0.35 * (np.arange(COLS) % 3 == 0))[None, :] + 0 * CX
    val[deck] = dark + np.array([1.0, 0.62, 0.25]) * (lamp[deck] * sweep[deck])[:, None]
    S.ch[m], S.val[m] = BR_CH[m], val[m]
    # traffic on the road deck: headlights one way, tail lights the other
    rd = int((BR_DECK - GY0) // CH)
    x = (CARS["x0"] + CARS["v"] * t) % (W + 200) - 100
    for xi, vi in zip(x, CARS["v"]):
        c = int(xi // CW)
        if 0 <= c < COLS:
            S.ch[rd, c] = "*" if vi > 0 else "."
            S.val[rd, c] = np.array([1.0, 0.95, 0.85]) * 0.7 if vi > 0 else np.array([1.0, 0.15, 0.1]) * 0.7
    # the Yurikamome train glides across on the lower deck
    rl = int((BR_LOW - GY0) // CH)
    front = -150 + (t - 9.5) / 7.5 * 1380
    for c in range(COLS):
        k = (front - CXs[c]) / CW
        if 0 <= k < 14:
            S.ch[rl, c] = "]" if k < 1 else "[" if k >= 13 else "|" if int(k) % 4 == 0 else "#"
            S.val[rl, c] = np.array([1.0, 0.95, 0.8]) * (0.85 if S.ch[rl, c] == "#" else 0.45)
    # and every lit part of the bridge drops a broken streak of light into the water below it
    rw = int((BR_WATER - GY0) // CH)
    bv = np.where(m[..., None], S.val, 0.0)
    cc = np.arange(COLS)
    for r in range(rw + 1, ROWS):
        rs = 2 * rw - r
        if rs < 0:
            break
        off = np.round(1.4 * np.sin(1.7 * t + 0.7 * r + 0.11 * cc)).astype(int)
        v = bv[rs, np.clip(cc + off, 0, COLS - 1)] * 0.4 * (0.4 + 0.6 * (0.5 + 0.5 * np.sin(3.1 * t + 1.3 * r + 0.7 * cc)))[:, None]
        more = v.max(-1) > S.val[r].max(-1)
        S.val[r, more] = v[more]
        S.ch[r, more & (v.max(-1) > 0.04)] = "~"
    return float(smooth((t - BRIDGE_ON) / 1.6))


# --- boats: a yakatabune drifts across with its paper lanterns lit, trailing a wake
def boats_layer(S, t):
    r = int((1200 - GY0) // CH)
    x0 = 90 + 21 * t
    for k in range(9):
        c = int((x0 + k * CW) // CW)
        if not 0 <= c < COLS:
            continue
        fl = 0.8 + 0.2 * math.sin(7.3 * t + 2.1 * k)
        S.ch[r - 1, c], S.val[r - 1, c] = "_", np.array([0.25, 0.15, 0.1]) * 0.25
        S.ch[r, c], S.val[r, c] = ("o" if k % 2 else "*"), np.array([1.0, 0.36, 0.12]) * 0.6 * fl
        S.ch[r + 1, c], S.val[r + 1, c] = "=", np.array([0.3, 0.16, 0.1]) * 0.2
        for j, a in ((2, 0.22), (3, 0.13), (4, 0.07)):
            S.ch[r + j, c], S.val[r + j, c] = ("|" if j < 4 else ":"), np.array([1.0, 0.4, 0.15]) * a * fl
    for k in range(1, 8):                          # the wake spreading behind
        c = int((x0 - k * CW) // CW)
        if 0 <= c < COLS:
            S.ch[r + 1, c] = "~"
            S.val[r + 1, c] = np.array([0.7, 0.75, 0.85]) * 0.12 * (1 - k / 8)


class Scene:
    def __init__(self):
        self.ch = np.full((ROWS, COLS), " ", "<U1")
        self.val = np.zeros((ROWS, COLS, 3))
        self.wash = np.zeros((ROWS, COLS, 3))


def compose(t):
    S = Scene()
    sky_layer(S, t)
    stars_layer(S, t)
    plane_layer(S, t)
    fuji_layer(S, t)
    tlit = float(smooth((t - TOWER_ON) / 1.4))
    windows = city_layer(S, t, tlit)
    neon_layer(S, t)
    tower = tower_layer(S, t)
    skytree = skytree_layer(S, t)
    bay_layer(S, t)
    bridge = bridge_layer(S, t)
    boats_layer(S, t)
    # light to characters: the brighter a cell, the nearer to white its colour runs
    m = S.val.max(-1)
    lum = 1 - np.exp(-1.6 * m)
    col = S.val / np.maximum(m, 1e-9)[..., None]
    rgb = col * (0.1 + 0.95 * lum)[..., None] + np.clip(lum - 0.78, 0, 1)[..., None] * 2.0
    rgb = (np.clip(rgb, 0, 1) * 255).astype(int)
    S.ch[(lum < 0.012) & (S.ch != " ")] = " "
    return S, rgb, dict(windows=windows, bridge=bridge, tower=tower, skytree=skytree)


# code panel: this very file scrolls by, following what is happening on screen
SRC = open(__file__, encoding="utf-8").read().splitlines()
NAME = os.path.basename(__file__)
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
PHASES = [(0.0, "sky"), (1.2, "fuji"), (2.4, "city"), (3.2, "neon"), (4.0, "bridge"), (6.0, "plane"), (8.2, "tower"),
          (11.7, "skytree"), (13.6, "bay"), (15.2, "boats"), (16.6, "stars"), (18.0, "render"), (19.2, "encode"), (DUR, "encode")]
KEYS = [(t, next(i for i, l in enumerate(SRC) if l.strip().startswith("# --- " + name + ":")))
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
    panel = Image.new("RGB", (PW, PH_), (14, 17, 26))
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
            d.rectangle((0, yy - 2, PW, yy + LH - 3), fill=(28, 35, 54))
        d.text((12, yy), f"{li + 1:>3}", font=F_CODE, fill=(75, 82, 100))
        x = 70
        for seg, kind in SRC_TOK[li]:
            seg = seg[: max(0, int((PW - 20 - x) / CHAR_W))]
            if seg:
                d.text((x, yy), seg, font=F_CODE, fill=TOK_COL[kind])
                x += len(seg) * CHAR_W
    d.rectangle((0, 0, PW, 52), fill=(14, 17, 26))
    for i, c in enumerate([(255, 95, 87), (254, 188, 46), (40, 200, 64)]):
        d.ellipse((20 + i * 26, 16, 36 + i * 26, 32), fill=c)
    d.text((110, 11), NAME, font=F_UI, fill=(200, 205, 215))
    d.ellipse((PW - 250, 18, PW - 236, 32), fill=(40, 200, 64))
    d.text((PW - 228, 11), "running  30 fps", font=F_UI, fill=(140, 150, 165))
    d.line((0, 46, PW, 46), fill=(34, 40, 56), width=2)
    mask = Image.new("L", (PW, PH_), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, PW - 1, PH_ - 1), 18, fill=255)
    img.paste(panel, (PX, PY), mask)
    ImageDraw.Draw(img).rounded_rectangle((PX, PY, PX + PW - 1, PY + PH_ - 1), 18, outline=(38, 46, 64), width=2)


# --- render: draw every lit cell over a faint wash of the sky, then let the lights bloom
BOX = (0, GY0 - 20, W, PY - 5)
BASE = Image.new("RGB", (W, H), (3, 4, 9))
_vy, _vx = np.mgrid[0:BOX[3] - BOX[1], 0:W]
_vr = np.hypot((_vx - W / 2) / (W * 0.66), (_vy - (BOX[3] - BOX[1]) / 2) / ((BOX[3] - BOX[1]) * 0.6))
VIGNETTE = Image.fromarray((255 * (1 - 0.4 * smooth(_vr - 0.3))).astype(np.uint8)).convert("RGB")


def frame(t):
    S, rgb, st = compose(t)
    wash = Image.fromarray(np.clip(S.wash * 120, 0, 255).astype(np.uint8)).resize((W, GH), Image.BICUBIC)
    wash = wash.filter(ImageFilter.GaussianBlur(18))
    txt = Image.new("RGB", (W, BOX[3] - BOX[1]))
    d = ImageDraw.Draw(txt)
    for r, c in np.argwhere(S.ch != " ").tolist():
        ch = S.ch[r, c]
        kana = ch >= "｡"                      # neon katakana come from a Japanese font, same cell
        d.text((c * CW, GY0 - BOX[1] + r * CH - (1 if kana else 0)), ch, font=F_KANA if kana else F_GRID,
               fill=tuple(rgb[r, c].tolist()))
    glow = txt.filter(ImageFilter.GaussianBlur(2.5)).point(lambda v: int(v * 0.5))
    halo = txt.filter(ImageFilter.GaussianBlur(12)).point(lambda v: min(255, int(v * 0.8)))
    hot = txt.point(lambda v: min(255, max(0, v - 110) * 2))           # only the brightest lights bloom wide
    wide = hot.filter(ImageFilter.GaussianBlur(28)).point(lambda v: min(255, int(v * 1.5)))
    scene = BASE.crop(BOX)
    sky = Image.new("RGB", scene.size)
    sky.paste(wash, (0, GY0 - BOX[1]))
    for layer in (sky, wide, halo, glow, txt):
        scene = ImageChops.add(scene, layer)
    scene = ImageChops.multiply(scene, VIGNETTE)
    img = BASE.copy()
    img.paste(scene, BOX[:2])

    d = ImageDraw.Draw(img)
    d.text((36, 34), NAME, font=F_UI, fill=(200, 205, 215))
    elev = round(-sun_dep(t), 1) + 0.0               # (+ 0.0 so it never reads "-0.0")
    d.text((36, 66), f"sun {elev:+.1f}\u00b0   windows {st['windows']:04d}", font=F_UI, fill=(120, 128, 145))
    onoff = lambda v: "on " if v > 0.5 else "off"
    d.text((36, 98), f"bridge {onoff(st['bridge'])}   tower {onoff(st['tower'])}   skytree {onoff(st['skytree'])}",
           font=F_UI, fill=(120, 128, 145))
    ts = f"t {t:05.2f}s"
    d.text((W - 36 - F_UI.getlength(ts), 34), ts, font=F_UI, fill=(120, 128, 145))
    code_panel(t, img)
    return img


# --- encode: pipe raw frames into ffmpeg, converted with the BT.709 matrix and tagged as such
def encode(out):
    ff = shutil.which("ffmpeg") or "ffmpeg"
    proc = subprocess.Popen(
        [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p,"
                "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv",
         "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-movflags", "+faststart", out],
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
        encode(sys.argv[1] if len(sys.argv) > 1 else "tokyo.mp4")
