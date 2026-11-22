"""Original BGM for volcano.mp4 (20 s), v2: heavy and oppressive, bass first.

C minor with a Phrygian Db. The bar grid is unchanged from v1 (1.82 s bars, so both
eruptions at 1.50 s and 10.60 s land on downbeats), but the feel is half-time and
slow: massive low hits on beats 1 and 3, and a chromatic C-Db grinding bass made
of heavily saturated C2 saws over two sub layers (C2 + C1). The low brass crawls
C-Db-C-Bb-B-C, with a dark "oh" choir and a tonal low drone. Lightning becomes
deep thunder, not a high stab. No spiccato, anvil, ticks, cymbals or high stabs,
and the master is low-passed so the whole piece stays dark.
"""
import sys, wave
import numpy as np

SR = 48000
DUR = 20.0
N = int(SR * DUR) + SR * 4
rng = np.random.default_rng(20261126)

BLAST1, BLAST2 = 1.50, 10.60
BAR = (BLAST2 - BLAST1) / 5          # 1.82 s
BEAT = BAR / 4
S16 = BAR / 16
LIGHTNING = [5.43, 5.97, 6.23, 11.30, 11.50, 13.77, 14.00, 14.20, 18.80, 18.97, 19.30]


def tbar(bar, p16=0.0):
    return BLAST1 + (bar - 1) * BAR + p16 * S16


def mtof(m):
    return 440.0 * 2.0 ** ((np.asarray(m, float) - 69.0) / 12.0)


def tt(T):
    return np.arange(int(T * SR)) / SR


def fft_filter(x, lo=None, hi=None, order=2):
    n = x.shape[-1]
    X = np.fft.rfft(x, axis=-1)
    f = np.fft.rfftfreq(n, 1 / SR)
    H = np.ones_like(f)
    if hi is not None:
        H /= np.sqrt(1 + (f / hi) ** (2 * order))
    if lo is not None:
        H /= np.sqrt(1 + (lo / np.maximum(f, 1e-3)) ** (2 * order))
    return np.fft.irfft(X * H, n, axis=-1)


def noise(n):
    return rng.standard_normal(n)


STEMS = {}
GAIN = dict(growl=0.42, sub=0.30, sub2=0.20, drone=0.12, hit=0.34, slam=0.45, tom=0.26, heart=0.36,
            brass=0.16, cluster=0.070, choir=0.10, impact=0.40, thunder=0.21, rise=0.09)
SEND = dict(growl=.05, sub=.0, sub2=.0, drone=.10, hit=.20, slam=.30, tom=.22, heart=.15,
            brass=.28, cluster=.35, choir=.45, impact=.22, thunder=.30, rise=.30)


def place(name, sig, t, pan=0.0, gain=1.0):
    buf = STEMS.setdefault(name, np.zeros((2, N), np.float32))
    i = int(round(t * SR))
    if sig.ndim == 1:
        th = (np.clip(pan, -1, 1) + 1) * np.pi / 4
        st = np.vstack([sig * np.cos(th), sig * np.sin(th)]) * np.sqrt(2)
    else:
        st = sig.copy()
    k = min(st.shape[1], max(int(0.01 * SR), int(0.04 * st.shape[1])))
    st[:, -k:] *= np.cos(np.linspace(0, np.pi / 2, k)) ** 2
    if i < 0:
        st, i = st[:, -i:], 0
    n = min(st.shape[1], N - i)
    if n > 0:
        buf[:, i:i + n] += (st[:, :n] * gain).astype(np.float32)


# ---------------------------------------------------------------- instruments
def saw(f, t, fc, kmax=40, ph0=0.0):
    out = np.zeros_like(t)
    for k in range(1, kmax + 1):
        if f * k > 9000:
            break
        out += np.sin(2 * np.pi * f * k * t + ph0 * k) / k / np.sqrt(1 + (f * k / fc) ** 4)
    return out


def growl(m, dur, vel=0.9):
    """Heavy grinding bass: three detuned saws, hard saturation, dark slowly-closing filter."""
    f = float(mtof(m)); T = dur + 0.1; t = tt(T)
    fc = 520 + 900 * np.exp(-t / 0.25) + 120 * np.sin(2 * np.pi * 1.1 * t)
    x = (saw(f * 2 ** (-11 / 1200), t, fc) + saw(f, t, fc, ph0=0.7) + saw(f * 2 ** (11 / 1200), t, fc, ph0=1.9))
    x = np.tanh(3.0 * x) / np.tanh(3.0)
    env = np.minimum(1, t / 0.012) * (0.85 + 0.15 * np.exp(-t / 0.3)) * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.04))
    return fft_filter(x, hi=2400) * env * vel          # keep the grit so the bass reads on small speakers


def subnote(m, dur, vel=0.9):
    f = float(mtof(m)); T = dur + 0.1; t = tt(T)
    env = np.minimum(1, t / 0.012) * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.05))
    return np.tanh(1.3 * np.sin(2 * np.pi * f * t)) * env * vel


def drone(m, T, vel=0.6):
    t = tt(T); f = float(mtof(m)); x = np.zeros_like(t)
    for det in (-7, 0, 6):
        x += saw(f * 2 ** (det / 1200), t, 260 + 80 * np.sin(2 * np.pi * 0.13 * t), kmax=10)
    return np.tanh(1.5 * x) * np.sin(np.minimum(1, t / 1.2) * np.pi / 2) ** 2 * vel


def hit(vel=1.0, f0=42):
    """Massive low drum hit (big taiko / cinematic boom drum), dark."""
    t = tt(1.8)
    ph = 2 * np.pi * np.cumsum(f0 * (1 + 1.2 * np.exp(-t / 0.035))) / SR
    body = np.sin(ph) * np.exp(-t / 0.6) + 0.3 * np.sin(1.5 * ph) * np.exp(-t / 0.2)
    nz = fft_filter(noise(len(t)), hi=500); nz /= nz.std()
    return np.tanh(1.4 * (body + 0.5 * nz * np.exp(-t / 0.03))) * np.minimum(1, t / 0.001) * vel


def slam(vel=1.0):
    """Half-time backbeat: a low, dark, roomy slam (tuned low, no bright snare wires)."""
    t = tt(1.0)
    nz = fft_filter(noise(len(t)), lo=120, hi=1800); nz /= nz.std()
    tone = np.sin(2 * np.pi * np.cumsum(110 * (1 + 0.3 * np.exp(-t / 0.02))) / SR) * np.exp(-t / 0.16)
    return np.tanh(0.8 * nz * np.exp(-t / 0.12) + tone) * np.minimum(1, t / 0.001) * vel


def tom(f0, vel=0.7):
    t = tt(0.9)
    ph = 2 * np.pi * np.cumsum(f0 * (1 + 0.5 * np.exp(-t / 0.03))) / SR
    return np.sin(ph) * np.exp(-t / 0.3) * vel


def heart(vel=0.7):
    """Low heartbeat thud for the oppressive lull."""
    t = tt(0.6)
    return np.sin(2 * np.pi * np.cumsum(48 * (1 + 0.6 * np.exp(-t / 0.02))) / SR) * np.exp(-t / 0.14) * vel


def brass(notes, dur, vel=0.8, attack=0.12, bright=1400):
    """Low brass (trombones + tuba), filtered dark."""
    T = dur + 0.6; t = tt(T)
    fc = 250 + bright * (1 - np.exp(-t / 0.2)) * np.exp(-np.maximum(t - 0.5, 0) / 2.0)
    L = np.zeros_like(t); R = np.zeros_like(t)
    for m in notes:
        for det, wl in [(-6, .8), (0, .5), (6, .2)]:
            s = saw(float(mtof(m)) * 2 ** (det / 1200), t, fc, kmax=24, ph0=rng.uniform(0, 6))
            L += s * np.sqrt(wl); R += s * np.sqrt(1 - wl)
    env = np.minimum(1, t / attack) * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.3))
    return np.vstack([np.tanh(L), np.tanh(R)]) * env * vel


VOWEL_OH = [(480, 80, 1.0), (820, 110, 0.45), (2500, 200, 0.05)]


def choir(m, dur, attack=0.8):
    f0 = float(mtof(m)); T = dur + 1.5; t = tt(T)
    L = np.zeros_like(t); R = np.zeros_like(t)
    form = lambda f: sum(a / (1 + ((f - F) / bw) ** 2) for F, bw, a in VOWEL_OH) + 0.01
    for v in range(4):
        vib = 12 * np.sin(2 * np.pi * rng.uniform(4.3, 5.2) * t + rng.uniform(0, 6)) + rng.normal(0, 7)
        ph = 2 * np.pi * np.cumsum(f0 * 2 ** (vib / 1200)) / SR
        s = sum(form(f0 * k) / k ** 0.6 * np.sin(k * ph + rng.uniform(0, 6)) for k in range(1, 30) if f0 * k < 3200)
        w = (v + 0.5) / 4
        L += s * np.sqrt(w); R += s * np.sqrt(1 - w)
    env = np.sin(np.minimum(1, t / attack) * np.pi / 2) ** 2 * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.7))
    return np.vstack([L * env, R * env])


def subdrop(f0=80, f1=26, T=3.0, vel=1.0):
    t = tt(T)
    fr = f1 + (f0 - f1) * np.exp(-t / 0.45)
    return np.tanh(1.8 * np.sin(2 * np.pi * np.cumsum(fr) / SR)) * np.minimum(1, t / 0.003) * np.exp(-t / 1.3) * vel


def boom(vel=1.0, f0=34):
    t = tt(2.8)
    nz = fft_filter(noise(len(t)), hi=350); nz /= nz.std()
    body = np.sin(2 * np.pi * np.cumsum(f0 + 50 * np.exp(-t / 0.08)) / SR)
    return np.tanh(1.3 * (body * np.exp(-t / 0.9) + 0.5 * nz * np.exp(-t / 0.22))) * np.minimum(1, t / 0.002) * vel


def low_rise(T, m0=24, m1=36, vel=0.6):
    """Tonal low riser: the bass drone bends up an octave and swells into the hit."""
    t = tt(T); m = m0 + (m1 - m0) * (t / T) ** 1.8
    ph = 2 * np.pi * np.cumsum(mtof(m)) / SR
    x = sum(np.sin(k * ph) / k for k in range(1, 9))
    return np.tanh(1.5 * x) * (t / T) ** 2.2 * vel


# ---------------------------------------------------------------- harmony
PROG = {1: ['Cm', 'Cm'], 2: ['Cm', 'Db'], 3: ['Ab', 'G'], 4: ['Db', 'Db'], 5: ['Db', 'G'],
        6: ['Cm', 'Cm'], 7: ['Ab', 'Db'], 8: ['Bbm', 'G'], 9: ['Cm', 'Cm'], 10: ['Db', 'Cm']}
ROOT = {'Cm': 36, 'Db': 37, 'Ab': 32, 'G': 31, 'Bbm': 34}        # bass register C2 area
CLUSTER = {'Cm': [48, 51, 55], 'Db': [49, 53, 56], 'Ab': [44, 48, 51], 'G': [43, 47, 50], 'Bbm': [46, 49, 53]}

# -- tonal drone under everything
place('drone', drone(24, 20.5, 0.8), 0.0)

# -- intro: two heartbeats, low riser into the first eruption
for t0, v in [(0.25, .6), (0.45, .45), (0.95, .75), (1.15, .55)]:
    place('heart', heart(v), t0)
place('rise', low_rise(1.45, 24, 36, 0.9), BLAST1 - 1.45)
place('cluster', brass([36, 37, 43], 1.45, 0.6, attack=1.2, bright=500), BLAST1 - 1.45)

# -- eruptions: sub drop + boom + huge drum + dark low-brass cluster + choir
for tb_, big in [(BLAST1, 0.9), (BLAST2, 1.0)]:
    place('impact', subdrop(vel=big), tb_)
    place('impact', boom(big), tb_)
    place('hit', hit(1.2 * big, 38), tb_)
    place('brass', brass([24, 36, 37, 43], 2.2, 1.0 * big, attack=0.03, bright=1100), tb_)
    place('choir', choir(48, 2.0, attack=0.15), tb_, gain=big)
    place('choir', choir(49, 2.0, attack=0.15), tb_, gain=0.7 * big)

# -- grinding bass: R (dotted half feel) then the chromatic b2 rub, half-time
GRIND = [(0, 0, 6), (6, 0, 2), (8, 1, 4), (12, 0, 4)]


def bass(m, d, gv, t0):
    place('growl', growl(m, d, gv), t0)
    place('sub', subnote(m, d, 0.9), t0)
    if m - 12 >= 24:                                   # C1 / Db1 sub layer; nothing below ~32 Hz
        place('sub2', subnote(m - 12, d, 0.9), t0)


for bar in range(1, 11):
    for half in (0, 1):
        ch = PROG[bar][half]; root = ROOT[ch]
        if bar in (1, 2, 3, 6, 7, 8):
            for p, iv, ln in GRIND:
                if (p >= 8) != bool(half):
                    continue
                m = root + iv
                d = ln * S16 * 0.94
                bass(m, d, 0.95 if p in (0, 8) else 0.75, tbar(bar, p))
        else:
            ln = BAR / 2 * 0.98 if not (bar == 10 and half == 1) else DUR - tbar(10, 8)
            gv = {4: 0.42, 5: 0.55 + 0.1 * half, 9: 0.8, 10: 0.75}[bar]
            bass(root, ln, gv, tbar(bar, half * 8))

# -- low brass cluster pads (oppressive sustain) and the crawling brass line over eruption 2
for bar in range(1, 11):
    for half in (0, 1):
        ch = PROG[bar][half]
        ln = BAR / 2 if not (bar == 10 and half == 1) else DUR - tbar(10, 8)
        lv = {4: 0.6, 5: 0.75}.get(bar, 0.9)
        place('cluster', brass(CLUSTER[ch], ln, lv, attack=0.35, bright=600), tbar(bar, half * 8))
CRAWL = [(6, 0, 48, 16), (7, 0, 48, 8), (7, 8, 49, 8), (8, 0, 46, 8), (8, 8, 47, 8), (9, 0, 48, 16)]
for bar, p, m, ln in CRAWL:
    d = ln * S16 * 0.97
    place('brass', brass([m, m - 12], d, 0.85, attack=0.15), tbar(bar, p))
    place('choir', choir(m, d, attack=0.3), tbar(bar, p), gain=0.75)
place('brass', brass([37, 49], BAR / 2, 0.85, attack=0.1), tbar(10, 0))
place('brass', brass([36, 48], DUR - tbar(10, 8), 0.8, attack=0.15), tbar(10, 8))
for bar in (4, 5):                      # the lull: a low choir hum swells under the heartbeat
    place('choir', choir(49, BAR, attack=1.0), tbar(bar), gain=0.55 if bar == 4 else 0.8)

# -- drums: half-time, few and massive
for bar in range(1, 11):
    if bar in (1, 2, 3):
        hits, slams, toms = [0, 14] if bar == 3 else [0], [8], ([(6, 70)] if bar >= 2 else [])
    elif bar in (6, 7, 8):
        hits, slams, toms = [0, 6, 10], [8], [(12, 75), (14, 60)]
    elif bar == 9:
        hits, slams, toms = [0], [8], []
    elif bar == 10:
        hits, slams, toms = [0], [], []
    else:
        hits, slams, toms = [], [], []
    for p in hits:
        if (bar, p) in [(1, 0), (6, 0)]:
            continue
        place('hit', hit(1.0 if p == 0 else 0.8), tbar(bar, p))
    for p in slams:
        place('slam', slam(0.9), tbar(bar, p), pan=-0.05)
    for p, f0 in toms:
        place('tom', tom(f0, 0.8), tbar(bar, p), pan=0.2)
# the lull: only a heartbeat (lub-dub), quickening in bar 5, then a low tom roll into eruption 2
for lub in (0, 8):
    place('heart', heart(0.8), tbar(4, lub)); place('heart', heart(0.55), tbar(4, lub + 1.5))
for p in (0, 4):
    place('heart', heart(0.85), tbar(5, p)); place('heart', heart(0.6), tbar(5, p + 1.5))
for k in range(8):
    place('tom', tom(62 + 3 * k, 0.35 + 0.08 * k), tbar(5, 8) + k * (BAR / 2) / 8, pan=0.1 - 0.03 * k)
place('rise', low_rise(BAR, 24, 36, 1.0), BLAST2 - BAR)

# -- lightning becomes deep thunder
for t0 in LIGHTNING:
    place('thunder', boom(0.6 if t0 < 15 else 0.45, f0=30), t0, pan=float(rng.uniform(-.3, .3)))


# ---------------------------------------------------------------- mix
def reverb(x2, T=3.0):
    n = int(T * SR); t = tt(T)
    irs = []
    for _ in range(2):
        nz = noise(n)
        w = np.exp(-t / 0.35)
        ir = (fft_filter(nz, hi=4500) * w + fft_filter(nz, hi=1500) * (1 - w)) * np.exp(-6.9 * t / T)
        ir = np.concatenate([np.zeros(int(0.03 * SR)), ir])
        irs.append(ir / np.sqrt((ir ** 2).sum()))
    M = 1 << int(np.ceil(np.log2(N + len(irs[0]))))
    out = np.zeros((2, N))
    for c in range(2):
        out[c] = np.fft.irfft(np.fft.rfft(x2[c], M) * np.fft.rfft(irs[c], M), M)[:N]
    return out


tN = np.arange(N) / SR
denv = np.interp(tN, [0, BLAST1, BLAST1 + 0.5, tbar(4), tbar(4) + 0.6, BLAST2, BLAST2 + 0.5, tbar(9), 20, 30],
                 [1.0, 1.0, 0.6, 0.6, 1.0, 1.0, 0.6, 0.6, 0.9, 0.9])
STEMS['drone'] = (fft_filter(STEMS['drone'].astype(float), lo=28, hi=380) * denv).astype(np.float32)
STEMS['sub'] = fft_filter(STEMS['sub'].astype(float), hi=160).astype(np.float32)
STEMS['sub2'] = fft_filter(STEMS['sub2'].astype(float), hi=90).astype(np.float32)
STEMS['choir'] = fft_filter(STEMS['choir'].astype(float), lo=90, hi=2200).astype(np.float32)
STEMS['cluster'] = fft_filter(STEMS['cluster'].astype(float), lo=60, hi=1200).astype(np.float32)
STEMS['brass'] = fft_filter(STEMS['brass'].astype(float), hi=2600).astype(np.float32)

# pre-impact "suck-out": everything except the impact drops away for ~0.12 s right before each eruption
suck = np.ones(N)
for tb_ in (BLAST1, BLAST2):
    suck = np.minimum(suck, np.interp(tN, [tb_ - 0.16, tb_ - 0.10, tb_ - 0.004, tb_], [1.0, 0.18, 0.18, 1.0]))
for k in STEMS:
    if k not in ('impact', 'hit', 'brass', 'choir'):
        STEMS[k] = (STEMS[k].astype(float) * suck).astype(np.float32)

dry = np.zeros((2, N)); snd = np.zeros((2, N))
WIN = dict(intro=(0, BLAST1), A=(BLAST1, tbar(4)), lull=(tbar(4), BLAST2), B=(BLAST2, tbar(9)), outro=(tbar(9), 20))
print('stem RMS dBFS per section:   ' + ' '.join(f'{w:>7s}' for w in WIN) + '    peak')
db = lambda v: 20 * np.log10(v + 1e-12)
for k, v in STEMS.items():
    x = v.astype(float) * GAIN[k]
    dry += x; snd += x * SEND[k]
    row = ' '.join(f'{db(np.sqrt((x[:, int(a*SR):int(b*SR)] ** 2).mean())):7.1f}' for a, b in WIN.values())
    print(f'  {k:8s}                  {row}   {db(np.abs(x).max()):6.1f}')

mix = dry + 0.22 * reverb(snd)
mix = fft_filter(mix, lo=22, hi=6500)            # keep the whole piece dark
mix = mix[:, :int(DUR * SR)]
t = np.arange(mix.shape[1]) / SR
fade = np.where(t < 19.2, 1.0, np.cos(np.clip((t - 19.2) / 0.8, 0, 1) * np.pi / 2) ** 2)
mix *= fade * np.minimum(1, t / 0.05)
mix /= np.abs(mix).max()
drive = 1.8
mix = np.tanh(drive * mix) / np.tanh(drive) * 0.89

out = sys.argv[1]
pcm = (np.clip(mix.T, -1, 1) * 32767).astype('<i2')
with wave.open(out, 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print('wrote', out, mix.shape[1] / SR, 's')
