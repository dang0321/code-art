"""Original BGM for aurora.mp4 (20 s): mysterious, beautiful, bright.

D major with Lydian colour (E/D, G#), celesta + harp + flute + 'ah' choir +
glockenspiel sparkles over a warm pad. 80 BPM 4/4 with bar 1 at 1.5 s, so the
bars land on the video's events: 7.5 s (first corona band), 10.5 s (second band
surges in), 13.5 s (brightest), 16.5 s (fading) -> final chord at 18.0 s.
"""
import sys, wave
import numpy as np

SR = 48000
DUR = 20.0
N = int(SR * DUR) + SR * 4
rng = np.random.default_rng(20261119)

BPM = 80.0
BEAT = 60.0 / BPM
T0 = 1.5


def tb(b):
    return T0 + b * BEAT


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


STEMS = {}
GAIN = dict(celesta=0.40, harp=0.29, flute=0.26, choir=0.075, pad=0.030, shimmer=0.040,
            bass=0.13, glock=0.19, timp=0.25, shaker=0.30, swell=0.06, wind=0.028)
SEND = dict(celesta=.40, harp=.35, flute=.40, choir=.55, pad=.45, shimmer=.60, bass=.05,
            glock=.60, timp=.25, shaker=.20, swell=.50, wind=.30)


def place(name, sig, t, pan=0.0, gain=1.0):
    buf = STEMS.setdefault(name, np.zeros((2, N), np.float32))
    i = int(round(t * SR))
    if sig.ndim == 1:
        th = (np.clip(pan, -1, 1) + 1) * np.pi / 4
        st = np.vstack([sig * np.cos(th), sig * np.sin(th)]) * np.sqrt(2)
    else:
        st = sig.copy()
    k = min(st.shape[1], max(int(0.02 * SR), int(0.05 * st.shape[1])))
    st[:, -k:] *= np.cos(np.linspace(0, np.pi / 2, k)) ** 2
    if i < 0:
        st, i = st[:, -i:], 0
    n = min(st.shape[1], N - i)
    if n > 0:
        buf[:, i:i + n] += (st[:, :n] * gain).astype(np.float32)


# ---------------------------------------------------------------- instruments
def celesta(m, vel=0.5):
    f = float(mtof(m)); T = float(np.interp(m, [60, 84, 100], [3.2, 2.2, 1.2])); t = tt(T)
    out = (np.sin(2 * np.pi * f * t) * np.exp(-t / (T * 0.30))
           + 0.22 * np.sin(2 * np.pi * 2 * f * t) * np.exp(-t / (T * 0.10))
           + 0.10 * np.sin(2 * np.pi * 3.98 * f * t) * np.exp(-t / (T * 0.04)))
    k = int(0.01 * SR)
    out[:k] += fft_filter(rng.standard_normal(k) * np.exp(-np.arange(k) / SR / 0.0015), lo=3000) * 0.06
    return out * np.minimum(1, t / 0.0012) * vel


def glock(m, vel=0.4):
    f = float(mtof(m)); t = tt(3.0); out = np.zeros_like(t)
    for r, d, a in zip([1, 2.76, 5.40], [1.4, 0.35, 0.12], [1, .35, .15]):
        if f * r < 16000:
            out += a * np.sin(2 * np.pi * f * r * t) * np.exp(-t / d)
    return out * np.minimum(1, t / 0.0008) * vel


def harp(m, vel=0.4):
    f = float(mtof(m)); T = float(np.interp(m, [40, 60, 84], [4.5, 3.2, 1.8])); t = tt(T)
    out = np.zeros_like(t)
    for n in range(1, 11):
        if f * n > 12000:
            break
        a = abs(np.sin(n * np.pi * 0.3)) / n ** 1.5
        out += a * np.sin(2 * np.pi * f * n * t) * np.exp(-t * (0.8 + 0.6 * n) * (f / 440) ** 0.3)
    return out * np.minimum(1, t / 0.002) * vel


def flute(m, dur, vel=0.5, swell=0.0, decay=None):
    f = float(mtof(m)); T = dur + 0.4; t = tt(T)
    cents = -25 * np.exp(-t / 0.04)
    vib = np.clip((t - 0.3) / 0.4, 0, 1) * np.sin(2 * np.pi * 5.0 * t) * 12
    ph = 2 * np.pi * np.cumsum(f * 2 ** ((cents + vib) / 1200)) / SR
    tone = np.sin(ph) + 0.22 * np.sin(2 * ph + 0.4) + 0.07 * np.sin(3 * ph + 1.1)
    env = np.minimum(1, t / 0.06) * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.08))
    env *= 1 + swell * np.clip(t / max(dur, 1e-3), 0, 1)
    if decay:
        env *= np.exp(-t / decay)
    nz = fft_filter(rng.standard_normal(len(t)), lo=f, hi=f * 4); nz /= nz.std() + 1e-9
    return (tone + 0.05 * nz) * env * vel


VOWEL = [(850, 110, 1.0), (1250, 130, 0.55), (2850, 180, 0.22), (3800, 220, 0.08)]   # soft 'ah'


def formant(f):
    return sum(a / (1 + ((f - F) / bw) ** 2) for F, bw, a in VOWEL) + 0.02


def choir(m, dur, voices=4, attack=0.6):
    f0 = float(mtof(m)); T = dur + 1.6; t = tt(T)
    L = np.zeros_like(t); R = np.zeros_like(t)
    for v in range(voices):
        det = rng.normal(0, 7)
        vib = 16 * np.sin(2 * np.pi * rng.uniform(4.6, 5.4) * t + rng.uniform(0, 6.28))
        ph = 2 * np.pi * np.cumsum(f0 * 2 ** ((det + vib) / 1200)) / SR
        s = np.zeros_like(t)
        for k in range(1, 40):
            if f0 * k > 6000:
                break
            s += formant(f0 * k) / k ** 0.7 * np.sin(k * ph + rng.uniform(0, 6.28))
        w = (v + 0.5) / voices
        L += s * np.sqrt(w); R += s * np.sqrt(1 - w)
    env = np.sin(np.minimum(1, t / attack) * np.pi / 2) ** 2
    env *= np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.45))
    return np.vstack([L * env, R * env])


def pad(notes, dur, attack=0.5):
    T = dur + 2.0; t = tt(T)
    L = np.zeros_like(t); R = np.zeros_like(t)
    for m in notes:
        for det, wl in [(-7, 0.85), (0, 0.5), (7, 0.15)]:
            f = float(mtof(m)) * 2 ** (det / 1200)
            s = np.zeros_like(t)
            for k in range(1, 9):
                if f * k > 9000:
                    break
                s += np.sin(2 * np.pi * f * k * t + rng.uniform(0, 2 * np.pi)) / k
            L += s * np.sqrt(wl); R += s * np.sqrt(1 - wl)
    env = np.sin(np.minimum(1, t / attack) * np.pi / 2) ** 2
    env *= np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.4))
    return np.vstack([L * env, R * env])


def shimmer(notes, dur):
    T = dur + 2.0; t = tt(T)
    L = np.zeros_like(t); R = np.zeros_like(t)
    for m in notes:
        f = float(mtof(m + 24))
        trem = 0.6 + 0.4 * np.sin(2 * np.pi * rng.uniform(0.3, 0.9) * t + rng.uniform(0, 6.28))
        s = np.sin(2 * np.pi * f * t) * trem
        w = rng.uniform(0.15, 0.85)
        L += s * np.sqrt(w); R += s * np.sqrt(1 - w)
    env = np.sin(np.minimum(1, t / 0.8) * np.pi / 2) ** 2 * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.6))
    return np.vstack([L * env, R * env])


def bass(m, dur, vel=0.5):
    f = float(mtof(m)); T = dur + 0.6; t = tt(T)
    ph = 2 * np.pi * f * t
    tone = np.sin(ph) + 0.30 * np.sin(2 * ph) + 0.08 * np.sin(3 * ph)
    env = np.minimum(1, t / 0.04) * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.15))
    return tone * env * vel


def timp(m, vel=0.6):
    f = float(mtof(m)); t = tt(2.0)
    ph = 2 * np.pi * np.cumsum(f * (1 + 0.02 * np.exp(-t / 0.05))) / SR
    body = np.sin(ph) * np.exp(-t / 0.9) + 0.35 * np.sin(1.5 * ph) * np.exp(-t / 0.4)
    nz = fft_filter(rng.standard_normal(len(t)), hi=500); nz /= nz.std()
    return (body + 0.25 * nz * np.exp(-t / 0.05)) * np.minimum(1, t / 0.003) * vel


def shaker(vel=0.4):
    t = tt(0.06)
    nz = fft_filter(rng.standard_normal(len(t)), lo=5000, hi=13000); nz /= nz.std()
    return nz * np.minimum(1, t / 0.004) * np.exp(-t / 0.016) * vel


def swell(T, vel=0.5):
    t = tt(T)
    nz = fft_filter(rng.standard_normal((2, len(t))), lo=1500, hi=10000); nz /= nz.std()
    return nz * (t / T) ** 3 * vel


# ---------------------------------------------------------------- score
CH = {
    'Dmaj9':    ([62, 66, 69, 73, 76], 38),
    'E/D':      ([64, 68, 71, 76], 38),
    'Bm9':      ([62, 66, 69, 73], 35),
    'Gmaj7#11': ([62, 66, 71, 73], 43),
    'Gmaj9':    ([62, 66, 69, 71], 43),
    'Aadd9':    ([64, 69, 71, 73], 45),
    'Bm7':      ([62, 66, 69, 71], 35),
    'Asus4':    ([64, 69, 74, 76], 45),
    'A':        ([64, 69, 73, 76], 45),
    'Gmaj7':    ([59, 62, 66, 71], 43),
    'Gm6':      ([58, 62, 64, 67], 43),
    'Dadd9':    ([62, 64, 66, 69, 74, 76], 38),
}
PROG = [(-2, 2, 'Dmaj9'), (0, 2, 'Dmaj9'), (2, 2, 'E/D'), (4, 2, 'Bm9'), (6, 2, 'Gmaj7#11'),
        (8, 2, 'Gmaj9'), (10, 2, 'Aadd9'), (12, 2, 'Dmaj9'), (14, 2, 'E/D'), (16, 2, 'Bm7'),
        (18, 1, 'Asus4'), (19, 1, 'A'), (20, 1, 'Gmaj7'), (21, 1, 'Gm6'), (22, 3.4, 'Dadd9')]


def lvl(b):   # follows the aurora's brightness
    return 0.60 if b < 0 else 0.55 if b < 4 else 0.70 if b < 8 else 0.85 if b < 12 else 1.0 if b < 20 else 0.8


for s, d, c in PROG:
    v, root = CH[c]
    dur = d * BEAT + 0.06
    place('pad', pad(v, dur, attack=1.3 if s < 0 else 0.4), tb(s), gain=lvl(s))
    place('shimmer', shimmer(v[:4], dur), tb(s), gain=1.0 if (s < 4 or s >= 20) else 0.6)
    if s >= 0:
        place('bass', bass(root, dur * 0.97, 0.5 * lvl(s) + 0.1), tb(s))
    if s >= 8:
        for m in v[:4]:
            place('choir', choir(m, dur, attack=0.9 if s == 8 else 0.35), tb(s), gain=lvl(s))

# harp arpeggios: 8ths in bars 1-2, 16ths in bars 3-5, slowing in bar 6, rolled final chord
for s, d, c in PROG:
    if s < 0 or s >= 22:
        continue
    v, root = CH[c]
    pool = sorted(set([n - 12 for n in v[:3]] + v[:4]))
    seq = pool + pool[-2:0:-1]
    step = 0.5 if (s < 8 or s >= 20) else 0.25
    for k in range(int(d / step)):
        m = seq[k % len(seq)]
        place('harp', harp(m, (0.45 if k == 0 else 0.32) * (0.6 + 0.4 * lvl(s))), tb(s + k * step),
              pan=float(np.clip((m - 62) / 30, -0.5, 0.5)))
for i, m in enumerate([50, 57, 62, 64, 66, 69, 74, 76, 78, 81, 86]):
    place('harp', harp(m, 0.40), tb(22) + i * 0.055, pan=float(np.clip((m - 66) / 30, -0.6, 0.6)))

MEL = [(0, 81, 1), (1, 88, 1), (2, 86, .5), (2.5, 80, .5), (3, 83, 1),
       (4, 85, 1), (5, 81, .5), (5.5, 78, .5), (6, 83, 1.5), (7.5, 85, .5),
       (8, 86, 1), (9, 90, 1), (10, 88, 1), (11, 85, .5), (11.5, 88, .5),
       (12, 86, .5), (12.5, 88, .5), (13, 90, 1), (14, 92, 1), (15, 88, .5), (15.5, 83, .5),
       (16, 93, 1.5), (17.5, 90, .5), (18, 88, 1), (19, 85, 1),
       (20, 83, 1), (21, 82, 1), (22, 81, 3)]
for b, m, d in MEL:
    place('celesta', celesta(m, 0.55 if b < 8 else 0.42), tb(b), pan=0.15)
    if b >= 8:
        last = b == 22
        place('flute', flute(m - 12, BEAT * d * (0.95 if not last else 1.0), 0.55 * lvl(b) + 0.1,
                             swell=0.12 if d >= 1 else 0, decay=1.4 if last else None), tb(b), pan=-0.1)

# glockenspiel stars: intro, at each band surge, and falling at the end
PENTA = [86, 88, 90, 93, 95, 98, 100]
for t, m, v in [(0.25, 93, .35), (0.70, 98, .25), (1.15, 90, .30),
                (7.55, 100, .30), (7.80, 95, .22), (10.50, 98, .35), (10.75, 93, .25),
                (13.50, 100, .35), (13.80, 95, .25),
                (18.25, 100, .30), (18.60, 98, .26), (18.95, 93, .22), (19.30, 90, .18), (19.65, 88, .14)]:
    place('glock', glock(m, v), t, pan=float(rng.uniform(-.7, .7)))
for _ in range(10):    # faint random twinkles in the quiet opening
    place('glock', glock(int(rng.choice(PENTA)), rng.uniform(.06, .14)), rng.uniform(0.0, 4.5),
          pan=float(rng.uniform(-.9, .9)))

# motion: shaker in the bright bars, soft timpani at the band surges, swells into them
for k in range(int((20 - 12) / 0.5)):
    b = 12 + k * 0.5
    place('shaker', shaker(0.55 if k % 2 == 0 else 0.35), tb(b), pan=0.4)
    place('shaker', shaker(0.20), tb(b + 0.25), pan=0.4)
for t, m, v in [(tb(8), 38, .55), (tb(12), 38, .9), (tb(16), 35, .8), (tb(22), 38, .6)]:
    place('timp', timp(m, v), t, pan=-0.05)
for t, T, v in [(tb(8), 1.5, .6), (tb(12), 2.2, 1.0), (tb(16), 1.5, .7)]:
    place('swell', swell(T, v), t - T)

# solar wind: band-passed air that follows the brightness curve of the video
LUM_T = np.arange(0, 20.01, 0.5)
LUM = np.array([21.6, 21.7, 21.6, 21.9, 23.0, 23.5, 24.0, 25.8, 26.9, 27.3, 28.1, 30.0, 31.5, 32.5, 32.6,
                30.9, 31.1, 33.1, 36.1, 36.1, 37.7, 44.2, 48.0, 48.6, 51.3, 52.3, 54.1, 57.2, 56.8, 53.9,
                50.7, 45.7, 39.8, 33.7, 29.6, 27.3, 24.5, 22.5, 21.0, 18.7, 18.0])
tw = tt(DUR)
wenv = 0.35 + 0.65 * (np.interp(tw, LUM_T, LUM) - 18) / (57.2 - 18)
wind = fft_filter(rng.standard_normal((2, len(tw))), lo=500, hi=2600)
wind /= wind.std()
drift = 0.5 + 0.5 * np.sin(2 * np.pi * 0.11 * tw)
wind = np.vstack([wind[0] * np.sqrt(drift), wind[1] * np.sqrt(1 - drift)]) * wenv
place('wind', wind, 0.0)


# ---------------------------------------------------------------- mix
def reverb(x2, T=4.0):
    n = int(T * SR); t = tt(T)
    irs = []
    for _ in range(2):
        nz = rng.standard_normal(n)
        w = np.exp(-t / 0.5)
        ir = (fft_filter(nz, hi=8000) * w + fft_filter(nz, hi=3000) * (1 - w)) * np.exp(-6.9 * t / T)
        ir = np.concatenate([np.zeros(int(0.03 * SR)), ir])
        irs.append(ir / np.sqrt((ir ** 2).sum()))
    M = 1 << int(np.ceil(np.log2(N + len(irs[0]))))
    out = np.zeros((2, N))
    for c in range(2):
        out[c] = np.fft.irfft(np.fft.rfft(x2[c], M) * np.fft.rfft(irs[c], M), M)[:N]
    return out


STEMS['pad'] = fft_filter(fft_filter(STEMS['pad'].astype(float), hi=2600), lo=150).astype(np.float32)
STEMS['choir'] = fft_filter(STEMS['choir'].astype(float), lo=200).astype(np.float32)

dry = np.zeros((2, N)); snd = np.zeros((2, N))
WIN = dict(intro=(0, T0), bar12=(T0, tb(8)), bar3=(tb(8), tb(12)), bar45=(tb(12), tb(20)), bar6=(tb(20), 20))
print('stem RMS dBFS per section:   ' + ' '.join(f'{w:>7s}' for w in WIN) + '    peak')
db = lambda v: 20 * np.log10(v + 1e-12)
for k, v in STEMS.items():
    x = v.astype(float) * GAIN[k]
    dry += x; snd += x * SEND[k]
    row = ' '.join(f'{db(np.sqrt((x[:, int(a*SR):int(b*SR)] ** 2).mean())):7.1f}' for a, b in WIN.values())
    print(f'  {k:8s}                  {row}   {db(np.abs(x).max()):6.1f}')

mix = dry + 0.38 * reverb(snd)
mix = fft_filter(mix, lo=45)
_X = np.fft.rfft(mix, axis=-1); _f = np.fft.rfftfreq(mix.shape[-1], 1 / SR)
_r = (_f / 5000) ** 2
mix = np.fft.irfft(_X * (1 + 0.41 * _r / (1 + _r)), mix.shape[-1], axis=-1)   # +3 dB air shelf
mix = mix[:, :int(DUR * SR)]
t = np.arange(mix.shape[1]) / SR
fade = np.where(t < 19.0, 1.0, np.cos(np.clip((t - 19.0) / 1.0, 0, 1) * np.pi / 2) ** 2)
mix *= fade * np.minimum(1, t / 0.3)
mix /= np.abs(mix).max()
drive = 1.2
mix = np.tanh(drive * mix) / np.tanh(drive) * 0.89

out = sys.argv[1]
pcm = (np.clip(mix.T, -1, 1) * 32767).astype('<i2')
with wave.open(out, 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print('wrote', out, mix.shape[1] / SR, 's')
