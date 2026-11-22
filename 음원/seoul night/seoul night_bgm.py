"""Original city-pop BGM for "seoul night.mp4" (20 s), v2.

Smooth midnight city pop (mood reference: late-70s Japanese city pop such as
"Stay With Me"; melody and progression are original). Less funk, same groove:
108 BPM so 9 bars fill exactly 20 s - 1 intro, 4 verse, 3 chorus, 1 landing bar.
Round finger bass, soft backbeat guitar chops, mellow FM e-piano (low index,
low-passed), strings counter-line, legato synth lead, 16th hats kept light.
"""
import sys, wave
import numpy as np

SR = 48000
DUR = 20.0
N = int(SR * DUR) + SR * 3
rng = np.random.default_rng(20261119)

BPM = 108.0
BEAT = 60.0 / BPM          # 0.5556 s, bar = 2.222 s, 9 bars = 20 s
T0 = 0.0


def tb(b):
    return T0 + b * BEAT


def t16(bar, pos):
    return tb((bar - 1) * 4 + pos / 4)


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
GAIN = dict(ep=0.19, bass=0.26, lead=0.30, guitar=0.30, strings=0.022, violins=0.060,
            kick=0.36, snare=0.34, hat=0.18, crash=0.08, tom=0.26, amb=0.020)
SEND = dict(ep=.25, bass=.02, lead=.30, guitar=.22, strings=.40, violins=.42,
            kick=.03, snare=.28, hat=.08, crash=.20, tom=.20, amb=.10)


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
def ep(m, dur, vel=0.5):
    """Mellow FM e-piano: low modulation index, faint tine (less bite than v1)."""
    f = float(mtof(m)); T = dur + 1.2; t = tt(T)
    idx = (0.6 + 1.6 * vel) * np.exp(-t / 0.2) + 0.25
    car = np.sin(2 * np.pi * f * t + idx * np.sin(2 * np.pi * f * t))
    tine = 0.15 * vel * np.sin(2 * np.pi * f * 7.0 * t) * np.exp(-t / 0.03)
    decay = float(np.interp(m, [48, 72, 96], [3.0, 1.9, 1.0]))
    env = np.minimum(1, t / 0.004) * np.exp(-t / decay) * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.12))
    return (car + tine) * env * vel


def bassnote(m, dur, vel=0.6):
    """Round finger bass."""
    f = float(mtof(m)); T = dur + 0.15; t = tt(T)
    out = np.zeros_like(t)
    for k in range(1, 9):
        if f * k > 5000:
            break
        a = 1.0 / k ** 1.6
        sus = 1.0 if k == 1 else 0.25 / k
        out += a * np.sin(2 * np.pi * f * k * t) * (sus + (1 - sus) * np.exp(-t * (3 + 4 * k)))
    env = np.minimum(1, t / 0.006) * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.04))
    return out * env * vel


def pluck(m, vel=0.5, T=0.45):
    f = float(mtof(m)); t = tt(T); out = np.zeros_like(t)
    for n in range(1, 11):
        if f * n > 9000:
            break
        out += abs(np.sin(n * np.pi * 0.2)) / n ** 1.3 * np.sin(2 * np.pi * f * n * t) * np.exp(-t * (6 + 2.2 * n))
    return out * np.minimum(1, t / 0.002) * vel


def strum(notes, vel=0.5, spread=0.012):
    T = 0.5 + spread * len(notes); out = np.zeros(int(T * SR))
    for i, m in enumerate(notes):
        s = pluck(m, vel * (0.8 + 0.2 * i / max(1, len(notes) - 1))); j = int(i * spread * SR)
        out[j:j + len(s)] += s[:len(out) - j]
    return out


def pad(notes, dur, attack=0.6):
    T = dur + 1.5; t = tt(T)
    L = np.zeros_like(t); R = np.zeros_like(t)
    for m in notes:
        for det, wl in [(-9, 0.85), (0, 0.5), (9, 0.15)]:
            f = float(mtof(m)) * 2 ** (det / 1200)
            s = np.zeros_like(t)
            for k in range(1, 9):
                if f * k > 9000:
                    break
                s += np.sin(2 * np.pi * f * k * t + rng.uniform(0, 6.28)) / k
            L += s * np.sqrt(wl); R += s * np.sqrt(1 - wl)
    env = np.sin(np.minimum(1, t / attack) * np.pi / 2) ** 2 * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.35))
    return np.vstack([L * env, R * env])


def violins(m, dur, vel=0.6, attack=0.18):
    """Section line: 5 detuned saw voices with vibrato, spread in stereo."""
    T = dur + 0.8; t = tt(T)
    L = np.zeros_like(t); R = np.zeros_like(t)
    for v in range(5):
        det = rng.normal(0, 6)
        vib = 12 * np.sin(2 * np.pi * rng.uniform(5.0, 6.0) * t + rng.uniform(0, 6.28)) * np.clip(t / 0.4, 0, 1)
        ph = 2 * np.pi * np.cumsum(float(mtof(m)) * 2 ** ((det + vib) / 1200)) / SR
        s = np.zeros_like(t)
        for k in range(1, 12):
            if float(mtof(m)) * k > 9000:
                break
            s += np.sin(k * ph) / k
        w = (v + 0.5) / 5
        L += s * np.sqrt(w); R += s * np.sqrt(1 - w)
    env = np.sin(np.minimum(1, t / attack) * np.pi / 2) ** 2 * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.25))
    return np.vstack([L * env, R * env]) * vel


def kick(vel=1.0):
    t = tt(0.45)
    ph = 2 * np.pi * np.cumsum(50 + 90 * np.exp(-t / 0.03)) / SR
    k = int(0.002 * SR)
    out = np.sin(ph) * np.exp(-t / 0.2)
    out[:k] += fft_filter(noise(k), lo=2000) * 0.12
    return out * vel


def snare(vel=1.0):
    t = tt(0.35)
    nz = fft_filter(noise(len(t)), lo=1200, hi=7000); nz /= nz.std()
    tone = np.sin(2 * np.pi * 185 * t) * np.exp(-t / 0.07) + 0.4 * np.sin(2 * np.pi * 320 * t) * np.exp(-t / 0.03)
    return (0.5 * nz * np.exp(-t / 0.13) + 0.8 * tone) * np.minimum(1, t / 0.001) * vel


def hat(vel=0.5, open_=False):
    t = tt(0.4 if open_ else 0.06)
    nz = fft_filter(noise(len(t)), lo=7000, hi=13000); nz /= nz.std()
    return nz * np.exp(-t / (0.11 if open_ else 0.016)) * np.minimum(1, t / 0.0008) * vel


def crash(vel=0.8):
    t = tt(2.4)
    nz = fft_filter(noise(len(t)), lo=4000, hi=12000); nz /= nz.std()
    metal = sum(np.sin(2 * np.pi * f * t + rng.uniform(0, 6)) for f in [3150, 4270, 5330, 6610]) / 4
    return (nz + 0.35 * metal) * np.exp(-t / 0.8) * np.minimum(1, t / 0.002) * vel


def tom(f0, vel=0.7):
    t = tt(0.5)
    ph = 2 * np.pi * np.cumsum(f0 * (1 + 0.4 * np.exp(-t / 0.03))) / SR
    return np.sin(ph) * np.exp(-t / 0.2) * vel


def mono_lead(notes, glide=0.06):
    L = int(DUR * SR) + SR
    pitch = np.full(L, np.nan); amp = np.zeros(L); vibe = np.zeros(L)
    for i, (on, off, m, v) in enumerate(notes):
        i0 = int(on * SR)
        nxt = int(notes[i + 1][0] * SR) if i + 1 < len(notes) else min(L, int((off + 1.0) * SR))
        end = min(nxt, int((off + 0.5) * SR))
        tl = np.arange(end - i0) / SR
        prev = notes[i - 1] if i > 0 else None
        legato = prev is not None and on - prev[1] < 0.03
        p = np.full(end - i0, float(m))
        if legato:
            g = min(len(p), int(glide * SR))
            p[:g] = m + (prev[2] - m) * np.exp(-np.arange(g) / (g / 4))
        pitch[i0:end] = p
        start = 0.7 if legato else 0.0
        env = start + (1 - start) * np.minimum(1, tl / (0.008 if legato else 0.02))
        env *= 0.88 + 0.12 * np.exp(-tl / 0.3)
        env *= np.where(tl < off - on, 1.0, np.exp(-(tl - (off - on)) / 0.08))
        amp[i0:end] = env * v
        vibe[i0:end] = np.clip((tl - 0.25) / 0.35, 0, 1)
    idx = np.where(~np.isnan(pitch), np.arange(L), 0); np.maximum.accumulate(idx, out=idx)
    pitch = pitch[idx]; pitch[np.isnan(pitch)] = 69
    t = np.arange(L) / SR
    fr = mtof(pitch) * 2 ** (vibe * 18 * np.sin(2 * np.pi * 5.4 * t) / 1200)
    ph = 2 * np.pi * np.cumsum(fr) / SR
    hk = [1, .42, .25, .14, .08, .05, .03, .02]
    tone = sum(a * np.sin((k + 1) * ph) for k, a in enumerate(hk))
    return (tone * amp)[:int(DUR * SR)]


# ---------------------------------------------------------------- harmony
CH = {
    'Bbmaj9': ([57, 60, 62, 65], 34),
    'A7b13':  ([55, 61, 65, 70], 33),
    'Dm9':    ([57, 60, 64, 65], 38),
    'Gm9':    ([58, 62, 65, 69], 43),
    'C9':     ([58, 62, 64, 67], 36),
    'Fmaj7':  ([60, 64, 65, 69], 41),
    'Bbmaj7': ([57, 62, 65, 70], 34),
    'Em7b5':  ([55, 58, 62, 64], 40),
    'Cm9':    ([58, 62, 63, 67], 36),
    'F9':     ([57, 60, 63, 67], 41),
    'Dm9end': ([57, 60, 64, 65, 69], 38),
}
PROG = [(0, 2, 'Bbmaj9'), (2, 2, 'A7b13'),                                    # intro
        (4, 4, 'Dm9'), (8, 2, 'Gm9'), (10, 2, 'C9'), (12, 2, 'Fmaj7'), (14, 2, 'Bbmaj7'),
        (16, 2, 'Em7b5'), (18, 2, 'A7b13'),                                    # verse
        (20, 2, 'Bbmaj9'), (22, 2, 'A7b13'), (24, 2, 'Dm9'), (26, 1, 'Cm9'), (27, 1, 'F9'),
        (28, 2, 'Bbmaj9'), (30, 1, 'Gm9'), (31, 1, 'A7b13'),                   # chorus
        (32, 4, 'Dm9end')]                                                     # landing
CHORUS = 20


def pan_of(m):
    return float(np.clip((m - 62) / 30, -0.45, 0.45))


# e-piano: fewer, longer hits (on 1 and 3, one soft push) instead of 16th syncopation
COMP = {4: [(0, 7, .48), (8, 5, .40), (14, 2, .32)], 2: [(0, 5, .46), (6, 2, .32)], 1: [(0, 3.6, .42)]}
for s, d, c in PROG:
    v, root = CH[c]
    if s >= 32:
        for m in v:
            place('ep', ep(m, 2.6, .48), tb(32), pan=pan_of(m))
        for i, m in enumerate([69, 72, 74, 77, 81]):
            place('ep', ep(m, 0.7, .30 - 0.03 * i), tb(33) + i * 0.14, pan=pan_of(m))
        continue
    pat = [(0, 7.5, .44)] if s < 4 else COMP[d]
    for pos, ln, vel in pat:
        for m in v:
            place('ep', ep(m, ln * BEAT / 4 * 0.97, vel), tb(s + pos / 4), pan=pan_of(m))

# bass: smooth, melodic, no slap / octave pops
BPAT = {
    4: [(0, 'R', 3), (3, 'R', 1), (4, 'F', 2), (6, 'O', 2), (8, 'R', 3), (11, 'R', 1), (12, 'F', 2), (14, 'A', 2)],
    2: [(0, 'R', 2.6), (3, 'R', 1), (4, 'F', 2), (6, 'A', 2)],
    1: [(0, 'R', 2.6), (3, 'A', 1)],
}
for i, (s, d, c) in enumerate(PROG):
    if s < 4:
        continue
    root = CH[c][1]
    if s >= 32:
        place('bass', bassnote(root, 2.4, .8), tb(32))
        continue
    nroot = CH[PROG[i + 1][2]][1]
    for pos, kind, ln in BPAT[d]:
        m = {'R': root, 'O': root + 12, 'F': root + 7,
             'A': nroot - 1 if nroot > root else nroot + 1}[kind]
        place('bass', bassnote(m, ln * BEAT / 4 * 0.9, .72 if pos == 0 else .58), tb(s + pos / 4))

# guitar: soft clean chords on 2 and 4 only
for bar in range(2, 9):
    for pos in (4, 12):
        t = t16(bar, pos)
        b = (bar - 1) * 4 + pos / 4
        c = [cc for ss, dd, cc in PROG if ss <= b < ss + dd][0]
        place('guitar', strum([n + 12 for n in CH[c][0]], .5), t, pan=0.6)

# strings bed from bar 4, counter-line in the chorus
for s, d, c in PROG:
    if s >= 12:
        v = [n + 12 for n in CH[c][0][:4]]
        place('strings', pad(v, d * BEAT + 0.05, attack=1.6 if s == 12 else 0.35), tb(s),
              gain=0.5 if s < CHORUS else 1.0)
for b, m, d in [(19.5, 81, 0.5), (20, 86, 2), (22, 85, 2), (24, 84, 2), (26, 82, 1), (27, 81, 1),
                (28, 86, 2), (30, 82, 1), (31, 85, 1), (32, 86, 3.6)]:
    place('violins', violins(m, d * BEAT * 0.98, .6 if b < 32 else .5, attack=0.12 if d < 1 else 0.22), tb(b))

# lead melody (absolute beat, length in beats, midi)
MEL = [(4.5, .5, 69), (5, .5, 72), (5.5, .75, 74), (6.25, .75, 76), (7, .5, 74), (7.5, .5, 72),
       (8, .75, 70), (8.75, .25, 69), (9, .5, 70), (9.5, 1.0, 74), (10.5, .5, 76), (11, .5, 74), (11.5, .5, 72),
       (12, .5, 72), (12.5, .75, 76), (13.25, .75, 77), (14, 1.0, 74), (15, .5, 77), (15.5, .5, 81),
       (16, .75, 79), (16.75, .25, 76), (17, .75, 74), (17.75, .25, 70), (18, 1.0, 73), (19, .5, 76),
       (19.5, .25, 77), (19.75, .25, 79),
       (20, 1.0, 81), (21, .5, 77), (21.5, .5, 79), (22, .5, 82), (22.5, .75, 81), (23.25, .25, 77), (23.5, .5, 76),
       (24, .75, 77), (24.75, .25, 76), (25, .5, 74), (25.5, .5, 69), (26, .75, 75), (26.75, .25, 74),
       (27, .5, 72), (27.5, .5, 75),
       (28, .5, 74), (28.5, .5, 77), (29, .75, 81), (29.75, 1.25, 84), (31, .5, 82), (31.5, .5, 81),
       (32, .5, 77), (32.5, .5, 76), (33, 2.6, 74)]
notes = []
for b, ln, m in MEL:
    on = tb(b); off = on + ln * BEAT * (0.93 if ln < 1 else 0.97)
    notes.append((on, off, m, 0.62 if b >= CHORUS else 0.52))
place('lead', mono_lead(notes), 0.0, pan=0.0)

# drums: steady and light - kick on 1 & 3 (verse) / four on the floor (chorus),
# snare on 2 & 4, 16th hats with soft ghosts, no ghost snares or claps
for bar in range(1, 10):
    chorus = bar >= 6
    if bar == 1:
        for p in range(8, 16, 2):
            place('hat', hat(.30 if p % 4 == 0 else .2), t16(1, p), pan=0.35)
        for p, v in [(14, .35), (15, .55)]:
            place('snare', snare(v), t16(1, p), pan=-0.05)
        continue
    if bar == 9:
        place('kick', kick(.95), t16(9, 0)); place('crash', crash(.8), t16(9, 0), pan=-0.3)
        continue
    kpos = [0, 4, 8, 12] if chorus else ([0, 8] if bar % 2 == 0 else [0, 8, 10])
    for p in kpos:
        place('kick', kick(.95 if p in (0, 8) else .8), t16(bar, p))
    for p in (4, 12):
        place('snare', snare(.85), t16(bar, p), pan=-0.05)
    fill = bar in (5, 8)
    for p in range(16):
        if fill and p >= 12:
            continue
        if p == 14 or (chorus and p % 4 == 2 and p != 14):
            place('hat', hat(.32 if chorus else .30, open_=True), t16(bar, p), pan=0.35)
        else:
            place('hat', hat([.45, .12, .30, .12][p % 4]), t16(bar, p), pan=0.35)
    if bar == 5:
        for p, v in [(12, .35), (13, .45), (14, .55), (15, .7)]:
            place('snare', snare(v), t16(5, p), pan=-0.05)
    if bar == 8:
        for p, f0, pn in [(12, 200, .3), (13, 165, .1), (14, 135, -.1), (15, 108, -.3)]:
            place('tom', tom(f0, .7), t16(8, p), pan=pn)
place('crash', crash(.65), t16(6, 0), pan=-0.3)

# night-city ambience: low hum + a distant car passing in the intro
ta = tt(DUR)
hum = fft_filter(noise(len(ta)), lo=60, hi=380); hum /= hum.std()
amb_env = np.where(ta < 2.2, 0.6, 0.3)
place('amb', np.vstack([hum, np.roll(hum, 3000)]) * amb_env, 0.0)
tc = tt(3.0)
car = fft_filter(noise(len(tc)), lo=200, hi=1800); car /= car.std()
ce = np.exp(-((tc - 1.2) / 0.55) ** 2) * 1.8
pp = np.clip(tc / 3.0, 0, 1)
place('amb', np.vstack([car * ce * np.sqrt(1 - pp), car * ce * np.sqrt(pp)]), 0.0)


# ---------------------------------------------------------------- mix
def reverb(x2, T=2.0):
    n = int(T * SR); t = tt(T)
    irs = []
    for _ in range(2):
        nz = noise(n)
        w = np.exp(-t / 0.3)
        ir = (fft_filter(nz, hi=8000) * w + fft_filter(nz, hi=3000) * (1 - w)) * np.exp(-6.9 * t / T)
        ir = np.concatenate([np.zeros(int(0.018 * SR)), ir])
        irs.append(ir / np.sqrt((ir ** 2).sum()))
    M = 1 << int(np.ceil(np.log2(N + len(irs[0]))))
    out = np.zeros((2, N))
    for c in range(2):
        out[c] = np.fft.irfft(np.fft.rfft(x2[c], M) * np.fft.rfft(irs[c], M), M)[:N]
    return out


tN = np.arange(N) / SR
trem = 0.14 * np.sin(2 * np.pi * 3.0 * tN)
STEMS['ep'] = (fft_filter(fft_filter(STEMS['ep'].astype(float), lo=110), hi=5000)
               * np.vstack([1 + trem, 1 - trem])).astype(np.float32)
STEMS['strings'] = fft_filter(fft_filter(STEMS['strings'].astype(float), hi=3200), lo=200).astype(np.float32)
STEMS['violins'] = fft_filter(fft_filter(STEMS['violins'].astype(float), hi=4500), lo=300).astype(np.float32)
STEMS['guitar'] = fft_filter(STEMS['guitar'].astype(float), lo=180, hi=6000).astype(np.float32)

dry = np.zeros((2, N)); snd = np.zeros((2, N))
WIN = dict(intro=(0, tb(4)), verse=(tb(4), tb(20)), chorus=(tb(20), tb(32)), end=(tb(32), 20))
print('stem RMS dBFS per section:   ' + ' '.join(f'{w:>7s}' for w in WIN) + '    peak')
db = lambda v: 20 * np.log10(v + 1e-12)
for k, v in STEMS.items():
    x = v.astype(float) * GAIN[k]
    dry += x; snd += x * SEND[k]
    row = ' '.join(f'{db(np.sqrt((x[:, int(a*SR):int(b*SR)] ** 2).mean())):7.1f}' for a, b in WIN.values())
    print(f'  {k:8s}                  {row}   {db(np.abs(x).max()):6.1f}')

mix = dry + 0.26 * reverb(snd)
mix = fft_filter(mix, lo=32)
mix = mix[:, :int(DUR * SR)]
t = np.arange(mix.shape[1]) / SR
fade = np.where(t < 19.0, 1.0, np.cos(np.clip((t - 19.0) / 1.0, 0, 1) * np.pi / 2) ** 2)
mix *= fade * np.minimum(1, t / 0.02)
mix /= np.abs(mix).max()
drive = 1.3
mix = np.tanh(drive * mix) / np.tanh(drive) * 0.89

out = sys.argv[1]
pcm = (np.clip(mix.T, -1, 1) * 32767).astype('<i2')
with wave.open(out, 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print('wrote', out, mix.shape[1] / SR, 's')
