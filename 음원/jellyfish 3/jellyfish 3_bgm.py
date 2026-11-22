"""Original BGM for "jellyfish 3.mp4" (20 s), v2: more mysterious, less playful.

A softly glowing jellyfish in the deep sea. There is no fixed tempo: every bell
contraction in the video (the HUD "beats" counter) is a bar downbeat, so the
music breathes with the jellyfish. Each contraction gets a deep low pulse and a
pad swell. Harmony sits on an F# pedal with Lydian #11 and a bII colour, and it
never fully resolves. A lone gliding glass-harmonica line has long, hazy echoes.
Slow crystal swells answer each glow, singing bowls mark the phrases, and a
far-off whale-like glide sits in the distance. All layers are tonal (no noise
beds). Removed from v1: vibraphone, harp arpeggios, bubbles, marimba mallet.
"""
import sys, wave
import numpy as np

SR = 48000
DUR = 20.0
N = int(SR * DUR) + SR * 7
rng = np.random.default_rng(20261122)

BEATS = [0.93, 3.80, 6.43, 9.17, 11.67, 14.47, 17.20, 19.80]     # bell contractions in the video


def tb(bar, pos=0.0):
    """bar 1..7 spans BEATS[bar-1] .. BEATS[bar]; pos in quarter beats 0..4."""
    a, b = BEATS[bar - 1], BEATS[bar]
    return a + (b - a) * pos / 4


def blen(bar):
    return BEATS[bar] - BEATS[bar - 1]


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
GAIN = dict(pad=0.030, low=0.032, choir=0.045, pulse=0.34, felt=0.26, lead=0.30, crystal=0.16,
            bowl=0.14, whale=0.12)
SEND = dict(pad=.55, low=.25, choir=.65, pulse=.15, felt=.30, lead=.55, crystal=.75, bowl=.60, whale=.85)


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
def pad(notes, dur, attack=1.0):
    T = dur + 3.0; t = tt(T)
    L = np.zeros_like(t); R = np.zeros_like(t)
    for m in notes:
        for det, wl in [(-7, 0.85), (0, 0.5), (7, 0.15)]:
            f = float(mtof(m)) * 2 ** (det / 1200)
            s = np.zeros_like(t)
            for k in range(1, 8):
                if f * k > 7000:
                    break
                s += np.sin(2 * np.pi * f * k * t + rng.uniform(0, 6.28)) / k ** 1.3
            L += s * np.sqrt(wl); R += s * np.sqrt(1 - wl)
    env = np.sin(np.minimum(1, t / attack) * np.pi / 2) ** 2 * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.9))
    return np.vstack([L * env, R * env])


VOWEL_OO = [(320, 70, 1.0), (800, 110, 0.40), (2300, 180, 0.06)]


def formant(f):
    return sum(a / (1 + ((f - F) / bw) ** 2) for F, bw, a in VOWEL_OO) + 0.01


def choir(m, dur, attack=1.4):
    f0 = float(mtof(m)); T = dur + 3.0; t = tt(T)
    L = np.zeros_like(t); R = np.zeros_like(t)
    for v in range(4):
        det = rng.normal(0, 6)
        vib = 12 * np.sin(2 * np.pi * rng.uniform(4.2, 5.0) * t + rng.uniform(0, 6.28))
        ph = 2 * np.pi * np.cumsum(f0 * 2 ** ((det + vib) / 1200)) / SR
        s = np.zeros_like(t)
        for k in range(1, 30):
            if f0 * k > 4000:
                break
            s += formant(f0 * k) / k ** 0.6 * np.sin(k * ph + rng.uniform(0, 6.28))
        w = (v + 0.5) / 4
        L += s * np.sqrt(w); R += s * np.sqrt(1 - w)
    env = np.sin(np.minimum(1, t / attack) * np.pi / 2) ** 2 * np.where(t < dur, 1.0, np.exp(-(t - dur) / 1.0))
    return np.vstack([L * env, R * env])


def pulse(m, vel=0.7):
    """The bell contraction: a deep, soft swell-and-release low tone."""
    f = float(mtof(m)); t = tt(3.2)
    ph = 2 * np.pi * np.cumsum(f * (1 - 0.03 * np.exp(-t / 0.15))) / SR
    tone = np.sin(ph) + 0.40 * np.sin(2 * ph) + 0.10 * np.sin(3 * ph)
    env = (1 - np.exp(-t / 0.08)) * np.exp(-t / 1.2)
    return tone * env * vel


def felt(m, vel=0.5):
    """Dark felted low note: defines the pulse without a bright, playful attack."""
    f = float(mtof(m)); t = tt(2.5)
    out = np.sin(2 * np.pi * f * t) * np.exp(-t / 1.1) + 0.30 * np.sin(2 * np.pi * 2 * f * t) * np.exp(-t / 0.4)
    return out * np.minimum(1, t / 0.012) * vel


def glide_line(notes, glide=0.16):
    """Lone glass-harmonica voice: one continuous tone that slides between notes."""
    L = int(DUR * SR) + SR
    pitch = np.full(L, np.nan); amp = np.zeros(L); vibe = np.zeros(L)
    for i, (on, off, m, v) in enumerate(notes):
        i0 = int(on * SR)
        nxt = int(notes[i + 1][0] * SR) if i + 1 < len(notes) else L
        end = min(nxt, L)
        tl = np.arange(end - i0) / SR
        prev = notes[i - 1] if i > 0 else None
        legato = prev is not None and on - prev[1] < 0.05
        p = np.full(end - i0, float(m))
        if legato:
            g = min(len(p), int(glide * SR))
            p[:g] = m + (prev[2] - m) * (0.5 + 0.5 * np.cos(np.linspace(0, np.pi, g)))
        pitch[i0:end] = p
        start = 0.85 if legato else 0.0
        env = start + (1 - start) * np.sin(np.minimum(1, tl / 0.35) * np.pi / 2) ** 2
        env *= np.where(tl < off - on, 1.0, np.exp(-(tl - (off - on)) / 0.5))
        amp[i0:end] = env * v
        vibe[i0:end] = np.clip((tl - 0.5) / 0.8, 0, 1)
    idx = np.where(~np.isnan(pitch), np.arange(L), 0); np.maximum.accumulate(idx, out=idx)
    pitch = pitch[idx]; pitch[np.isnan(pitch)] = 69
    t = np.arange(L) / SR
    fr = mtof(pitch) * 2 ** (vibe * 8 * np.sin(2 * np.pi * 4.3 * t) / 1200)
    ph = 2 * np.pi * np.cumsum(fr) / SR
    tone = np.sin(ph) + 0.10 * np.sin(2 * ph) + 0.04 * np.sin(3 * ph)
    return (tone * amp)[:int(DUR * SR)]


def crystal(m, vel=0.3, T=2.6):
    """Bowed-glass swell: glows in slowly, instead of twinkling."""
    f = float(mtof(m)); t = tt(T)
    tone = np.sin(2 * np.pi * f * t) + 0.15 * np.sin(2 * np.pi * 2.0 * f * t + 0.7)
    env = np.sin(np.minimum(1, t / 0.45) * np.pi / 2) ** 2 * np.exp(-np.maximum(t - 0.45, 0) / 0.9)
    return tone * env * vel


def bowl(f0, vel=0.4):
    """Singing bowl: inharmonic partials in close pairs that beat slowly."""
    t = tt(7.0); out = np.zeros_like(t)
    for r, d, a in [(1, 5.5, 1), (2.71, 3.5, .45), (5.03, 2.0, .18), (7.9, 1.0, .07)]:
        for db in (-0.7, 0.7):
            out += a * np.sin(2 * np.pi * (f0 * r + db) * t + rng.uniform(0, 6.28)) * np.exp(-t / d)
    return out * np.minimum(1, t / 0.01) * vel * 0.5


def whale(m0, m1, T, vel=0.3):
    """A distant, slow tonal glide (whale-song-like), heavily reverbed."""
    t = tt(T + 0.5); x = np.clip(t / T, 0, 1)
    m = m0 + (m1 - m0) * (0.5 - 0.5 * np.cos(np.pi * x)) + 0.6 * np.sin(np.pi * x)
    fr = mtof(m) * 2 ** (10 * np.sin(2 * np.pi * 3.0 * t) / 1200)
    ph = 2 * np.pi * np.cumsum(fr) / SR
    tone = np.sin(ph) + 0.25 * np.sin(2 * ph)
    env = np.sin(np.clip(t / T, 0, 1) * np.pi) ** 1.5
    return tone * env * vel


# ---------------------------------------------------------------- harmony on an F# pedal
CH = {
    'F#m9':      [57, 61, 64, 68],              # A C# E G#
    'Dmaj9#11':  [54, 61, 64, 68],              # F# C# E G#  (over D)
    'Bm9/F#':    [57, 61, 62, 66],              # A C# D F#
    'Gmaj9#11':  [59, 61, 62, 66],              # B C# D F#   (over G: bII colour)
    'F#9sus4':   [59, 61, 64, 68, 71],          # B C# E G# B  (E/F#, luminous, unresolved)
    'F#m(add9)': [56, 61, 64, 68],              # G# C# E G#
}
BASS = {'F#m9': 42, 'Dmaj9#11': 38, 'Bm9/F#': 42, 'Gmaj9#11': 43, 'F#9sus4': 42, 'F#m(add9)': 42}
BARS = {1: 'F#m9', 2: 'Dmaj9#11', 3: 'Bm9/F#', 4: 'Gmaj9#11', 5: 'F#9sus4', 6: 'Dmaj9#11', 7: 'F#m(add9)'}
LVL = {0: 0.5, 1: 0.65, 2: 0.7, 3: 0.8, 4: 0.88, 5: 1.0, 6: 0.85, 7: 0.75}

place('pad', pad(CH['F#m9'], BEATS[0] + 0.2, attack=0.9), 0.0, gain=LVL[0])
for bar in range(1, 8):
    v = CH[BARS[bar]]; root = BASS[BARS[bar]]
    dur = blen(bar) + 0.1 if bar < 7 else (DUR - BEATS[6]) + 0.8
    place('pad', pad(v, dur, attack=0.6), tb(bar), gain=LVL[bar])
    place('low', pad([root, root + 7], dur, attack=0.7), tb(bar), gain=LVL[bar])
    if bar >= 2:
        voices = v[:4] if bar != 5 else v[:4] + [76]
        for m in voices:
            place('choir', choir(m, dur, attack=1.6 if bar == 2 else 0.7), tb(bar),
                  gain=LVL[bar] * (0.6 if bar == 2 else 1.25 if bar == 5 else 1.0))

# each bell contraction: deep pulse + dark felt note (presence); the one at 19.8 s is a faint last pulse
for i, t0 in enumerate(BEATS):
    bar = min(i + 1, 7)
    root = BASS[BARS[bar]]
    v = 0.25 if i == 7 else (0.9 if bar == 5 else 0.75)
    place('pulse', pulse(root - 12 if root - 12 >= 30 else root, v), t0)
    place('felt', felt(root, v * 0.6), t0, pan=-0.05)

# singing bowls mark the phrases
for t0, f0, v in [(BEATS[0], 277.2, .5), (BEATS[2], 246.9, .45), (BEATS[4], 370.0, .55), (BEATS[6], 277.2, .5)]:
    place('bowl', bowl(f0, v), t0 + 0.05, pan=float(rng.uniform(-.4, .4)))

# lead: a lone, gliding glass-harmonica line (long notes, slides, hazy echoes)
MEL = {1: [(1.0, 73, 3.0)],
       2: [(0.0, 76, 1.5), (1.5, 80, 2.5)],
       3: [(0.0, 78, 2.0), (2.0, 73, 2.0)],
       4: [(0.0, 74, 1.5), (1.5, 73, 2.5)],
       5: [(0.0, 80, 2.0), (2.0, 83, 2.0)],
       6: [(0.0, 81, 2.0), (2.0, 80, 2.0)],
       7: [(0.0, 78, 1.5), (1.5, 80, 6.0)]}
notes = []
for bar, ns in MEL.items():
    for p, m, d in ns:
        on = tb(bar, p)
        off = min(on + d * blen(bar) / 4 * 0.995, DUR - 0.3)
        notes.append((on, off, m, 0.50 * (1.12 if bar == 5 else 1.0)))
line = glide_line(notes)
place('lead', line, 0.0, pan=0.0)
for k, (dl, g, p) in enumerate([(0.62, 0.32, -0.4), (1.24, 0.16, 0.4), (1.86, 0.08, -0.3)]):
    place('lead', fft_filter(line, hi=2500 / (k + 1)), dl, pan=p, gain=g)

# slow crystal swells answering each glow (about 0.3 s after a contraction); a wider bloom at the big light surge
GLOW = {0: [97], 1: [92], 2: [97, 90], 3: [95], 4: [97, 92, 100, 88], 5: [92], 6: [97, 90]}
for i, t0 in enumerate(BEATS[:7]):
    for j, m in enumerate(GLOW[i]):
        place('crystal', crystal(m, 0.30 if i != 4 else 0.34, T=2.6 if i != 4 else 3.4), t0 + 0.28 + j * 0.22,
              pan=float(rng.uniform(-.6, .6)))

# a far-off whale-like call in the distance
place('whale', whale(61, 66, 2.2, 0.5), 7.2, pan=-0.35)
place('whale', whale(64, 59, 2.4, 0.45), 15.3, pan=0.35)


# ---------------------------------------------------------------- mix
def reverb(x2, T=6.0):
    n = int(T * SR); t = tt(T)
    irs = []
    for _ in range(2):
        nz = rng.standard_normal(n)
        w = np.exp(-t / 0.7)
        ir = (fft_filter(nz, hi=5000) * w + fft_filter(nz, hi=1800) * (1 - w)) * np.exp(-6.9 * t / T)
        ir = np.concatenate([np.zeros(int(0.05 * SR)), ir])
        irs.append(ir / np.sqrt((ir ** 2).sum()))
    M = 1 << int(np.ceil(np.log2(N + len(irs[0]))))
    out = np.zeros((2, N))
    for c in range(2):
        out[c] = np.fft.irfft(np.fft.rfft(x2[c], M) * np.fft.rfft(irs[c], M), M)[:N]
    return out


# the pad, choir and low body breathe with the bell: swell just after every contraction
tN = np.arange(N) / SR
breath = np.full(N, 0.75)
for t0 in BEATS:
    x = tN - t0
    breath += 0.35 * np.where(x > 0, (1 - np.exp(-x / 0.3)) * np.exp(-x / 1.1), 0)
for k in ('pad', 'choir', 'low'):
    STEMS[k] = (STEMS[k].astype(float) * breath).astype(np.float32)
STEMS['pad'] = fft_filter(fft_filter(STEMS['pad'].astype(float), hi=1900), lo=120).astype(np.float32)
STEMS['choir'] = fft_filter(STEMS['choir'].astype(float), lo=150, hi=3500).astype(np.float32)
STEMS['low'] = fft_filter(fft_filter(STEMS['low'].astype(float), hi=700), lo=60).astype(np.float32)

dry = np.zeros((2, N)); snd = np.zeros((2, N))
WIN = dict(intro=(0, BEATS[0]), b1_2=(BEATS[0], BEATS[2]), b3_4=(BEATS[2], BEATS[4]),
           b5=(BEATS[4], BEATS[5]), b6_7=(BEATS[5], 20))
print('stem RMS dBFS per section:   ' + ' '.join(f'{w:>7s}' for w in WIN) + '    peak')
db = lambda v: 20 * np.log10(v + 1e-12)
for k, v in STEMS.items():
    x = v.astype(float) * GAIN[k]
    dry += x; snd += x * SEND[k]
    row = ' '.join(f'{db(np.sqrt((x[:, int(a*SR):int(b*SR)] ** 2).mean())):7.1f}' for a, b in WIN.values())
    print(f'  {k:8s}                  {row}   {db(np.abs(x).max()):6.1f}')

mix = dry + 0.50 * reverb(snd)
mix = fft_filter(mix, lo=28)
mix = mix[:, :int(DUR * SR)]
t = np.arange(mix.shape[1]) / SR
fade = np.where(t < 18.6, 1.0, np.cos(np.clip((t - 18.6) / 1.4, 0, 1) * np.pi / 2) ** 2)
mix *= fade * np.minimum(1, t / 0.05)
mix /= np.abs(mix).max()
drive = 1.15
mix = np.tanh(drive * mix) / np.tanh(drive) * 0.89

out = sys.argv[1]
pcm = (np.clip(mix.T, -1, 1) * 32767).astype('<i2')
with wave.open(out, 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print('wrote', out, mix.shape[1] / SR, 's')
