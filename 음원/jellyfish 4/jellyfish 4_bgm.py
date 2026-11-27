"""Original BGM for "jellyfish 4.mp4" (20 s): low and warm where the earlier jellyfish themes were high
and glassy; still mysterious, but brighter, and this time it resolves.

The violet jellyfish's bell contractions are the bar lines, so the music breathes with it. The harmony
moves in D Lydian - I, II over I, vi, IV, with the raised fourth that sounds like wonder - over double
basses. A cello section sings the tune in its warm middle register, and a low piano rolls a chord upward
with every pulse, as the ring of light runs from crown to rim. The little blue-green jelly has a voice of
its own: a harp note on each of its strokes, climbing as it climbs and panned to where it swims. At 11.8 s
it brushes the tentacles: a mallet cymbal and the timpani swell into the touch, and a vibraphone answers
each wave of light. Then it flees in quick harp figures, and the piece settles on a bright, open D.

Every instrument is synthesised from how the real one makes its sound, then played into a concert hall.

    python "jellyfish 4_bgm.py"                           -> jellyfish 4_bgm.wav (named after this file)
    python "jellyfish 4_bgm.py" --mux silent.mp4 out.mp4  -> also put the music under the video
"""
import math, os, shutil, subprocess, sys, wave
import numpy as np

SR = 48000
DUR = 20.0
NS = int(SR * (DUR + 4))                  # room for the tails
TAU = math.tau
rng = np.random.default_rng(20261123)

# --- timing: read off the video
BEATS = [0.93, 3.77, 6.43, 9.15, 11.65, 14.45, 17.17, 19.80]          # the violet bell's contractions
VISIT = [0.39, 2.01, 3.74, 5.68, 7.23, 9.21, 10.93,                     # the little jelly's strokes...
         12.05, 12.72, 13.40, 14.15, 15.80, 17.55, 19.30]               # ...quick once it is startled
TOUCH = 11.8
FLASH = [(12.0 + 0.55 * k, 0.8 ** k) for k in range(6)]               # the waves of light after the touch
VISIT_X = [(0, 260), (3, 313), (9, 427), (11.8, 463), (16, 276), (19, 115), (20, 63)]   # where it swims on screen


def hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def curve(points):                         # a level that glides between (time, value) points
    ts, vs = zip(*points)
    return lambda t: np.interp(t, ts, vs)


def band(lo, hi, k=2):                    # a soft band-pass response
    return lambda f: 1 / (1 + (lo / np.maximum(f, 1)) ** (2 * k)) / (1 + (f / hi) ** (2 * k))


def coloured(n, resp):                    # white noise shaped by a frequency response
    m = 1 << max(1, (n - 1).bit_length())
    X = np.fft.rfft(rng.normal(size=m))
    s = np.fft.irfft(X * resp(np.fft.rfftfreq(m, 1 / SR)), m)[:n]
    return s / (np.sqrt((s ** 2).mean()) + 1e-12)


class Bus:
    def __init__(self, send):
        self.x, self.send = np.zeros((2, NS)), send

    def add(self, s, t, pan=0.0, g=1.0):
        i = int(round(t * SR))
        if i < 0:
            s, i = s[-i:], 0
        n = min(len(s), NS - i)
        if n > 0:
            a = (np.clip(pan, -1, 1) + 1) * math.pi / 4
            self.x[0, i:i + n] += s[:n] * math.cos(a) * g
            self.x[1, i:i + n] += s[:n] * math.sin(a) * g


# --- piano: inharmonic partials of a stiff string, three strings per key slightly out of tune with
# each other (the slow beating), a fast first decay and a long aftersound, a felt hammer's thump
def piano(m, vel, off):
    f0 = hz(m)
    tr = 0.1 if m > 50 else 0.25
    tau0 = 9.0 * 2 ** (-(m - 36) / 20)
    L = min(off + 5 * tr, 5 * tau0, 9.0)
    t = np.arange(int(L * SR)) / SR
    n = np.arange(1, 40)
    B = np.clip(0.00035 * 2 ** ((m - 60) / 18), 6e-5, 0.004)
    fn = f0 * n * np.sqrt(1 + B * n * n)
    n, fn = n[fn < 12000], fn[fn < 12000]
    a = n ** -(2.0 - 1.1 * vel) * (0.15 + np.abs(np.sin(math.pi * n / 7.2)))
    a = a / (1 + (fn / (1400 + 4200 * vel)) ** 2)
    if m < 48:
        a[0] *= 0.6
    tau = tau0 / (1 + (fn / 2200) ** 1.3 + 0.04 * (n - 1))
    beat = fn * 0.00045 * rng.uniform(0.4, 1.4, len(n))
    ph = rng.uniform(0, TAU, len(n))
    s = np.zeros(len(t))
    for i0 in range(0, len(t), 32768):
        T = t[i0:i0 + 32768, None]
        env = 0.62 * np.exp(-T / (0.22 * tau)) + 0.38 * np.exp(-T / tau)
        s[i0:i0 + 32768] = (a * env * (1 + 0.28 * np.cos(TAU * beat * T + ph)) * np.sin(TAU * fn * T + ph)).sum(1)
    s *= (1 - np.exp(-t / 0.0025)) * np.where(t > off, np.exp(-(t - off) / tr), 1.0)
    s /= math.sqrt((a ** 2).sum())
    k = int(0.04 * SR)
    s[:k] += coloured(k, band(60, 900 + 1500 * vel)) * np.exp(-np.arange(k) / SR / 0.007) * 0.04 * vel
    return s * vel ** 1.5


# --- strings: each section is several players, each a little out of tune with the rest, each with
# their own vibrato (fading in after the bow settles) and bow pressure; the wooden body colours it all
BODIES = {"va": [(230, 0.9, 0.35), (380, 1.0, 0.3), (850, 0.5, 0.5), (2200, 1.0, 0.6)],
          "vc": [(100, 0.8, 0.35), (200, 1.0, 0.3), (580, 0.6, 0.5), (1600, 0.8, 0.6)],
          "cb": [(60, 0.8, 0.4), (110, 1.0, 0.35), (300, 0.5, 0.5), (900, 0.5, 0.6)]}


def body(f, kind):
    g = 0.25 + sum(gn * np.exp(-0.5 * (np.log2(np.maximum(f, 1) / fc) / w) ** 2) for fc, gn, w in BODIES[kind])
    return g / (1 + (f / 6000) ** 4)


def bowed(m, t0, dur, dyn, kind="vc", players=6, glide=None, att=0.12, rel=0.35):
    f0 = hz(m)
    nS = int((dur + rel) * SR)
    t = np.arange(nS) / SR
    lvl = dyn(t0 + np.minimum(t, dur))
    env = lvl * smooth(t / att) * np.where(t > dur, np.exp(-(t - dur) / (rel / 3)), 1.0)
    n = np.arange(1, int(min(48, 11000 / f0)) + 1)
    bright = float(lvl.max())
    out = np.zeros(nS)
    for p in range(players):
        det = 2 ** (rng.normal(0, 5) / 1200)
        vib = rng.uniform(0.0022, 0.004) * smooth((t - 0.22) / 0.5) * np.sin(TAU * rng.uniform(5.0, 6.3) * t + rng.uniform(0, TAU))
        f = f0 * det * (1 + vib)
        if glide is not None:                              # a slide up or down into the note
            g = smooth(t / 0.12)
            f = f * (hz(glide) / f0) ** (1 - g)
        phase = TAU * np.cumsum(f) / SR
        fn = n * f0
        amp = body(fn, kind) / n / (1 + (fn / (1800 + 3800 * bright)) ** 2)
        wob = 1 + 0.08 * np.sin(TAU * rng.uniform(0.3, 0.9) * t + rng.uniform(0, TAU))
        late = int(abs(rng.normal(0, 0.012)) * SR)
        phs = rng.uniform(0, TAU, len(n))
        s = np.zeros(nS)
        for i0 in range(0, nS, 32768):
            s[i0:i0 + 32768] = np.sin(phase[i0:i0 + 32768, None] * n + phs) @ amp
        out[late:] += (s * wob)[:nS - late]
    out /= math.sqrt(players) * math.sqrt((amp ** 2).sum()) * 2.2
    rosin = coloured(nS, lambda fr: body(fr, kind) * band(200, 6000)(fr)) * 0.025
    return (out + rosin) * env


# --- harp: a plucked gut string - every harmonic it was plucked with, the high ones dying first;
# the same string plucked short and dark is a double bass pizzicato
def harp(m, vel, ring=1.0, dark=0.0):
    f0 = hz(m)
    L = ring * (2.2 + 3.0 * 2 ** (-(m - 50) / 12))
    t = np.arange(int(L * SR)) / SR
    n = np.arange(1, 30)
    fn = f0 * n * (1 + 0.00008 * n * n)
    n, fn = n[fn < 9000], fn[fn < 9000]
    a = np.abs(np.sin(math.pi * n * 0.13)) / n ** (1.25 + 0.6 * (1 - vel) + dark)   # plucked an eighth along
    tau = ring * (2.0 * 2 ** (-(m - 57) / 24) + 0.5) / (1 + (fn / (1400 - 700 * dark)) ** 1.4)
    ph = rng.uniform(0, TAU, len(n))
    s = np.zeros(len(t))
    for i0 in range(0, len(t), 32768):
        T = t[i0:i0 + 32768, None]
        s[i0:i0 + 32768] = (a * np.exp(-T / tau) * np.sin(TAU * fn * T + ph)).sum(1)
    s *= 1 - np.exp(-t / 0.0015)
    s /= math.sqrt((a ** 2).sum())
    k = int(0.02 * SR)
    s[:k] += coloured(k, band(400, 3500)) * np.exp(-np.arange(k) / SR / 0.004) * 0.05 * vel   # the finger leaving
    return s * vel


# --- percussion: membranes, bars and plates ring in their own inharmonic modes
def modes(f0, ratios, amps, decays, L, bend=0.0, bend_t=0.05):
    t = np.arange(int(L * SR)) / SR
    pitch = 1 + bend * np.exp(-t / bend_t)
    ph = TAU * np.cumsum(pitch) / SR
    return sum(a * np.exp(-t / d) * np.sin(ph * f0 * r) for r, a, d in zip(ratios, amps, decays)), t


def timpani(m, vel, L=3.5):
    s, t = modes(hz(m), [1.0, 1.5, 1.99, 2.44, 2.9], [1, 0.6, 0.35, 0.2, 0.12], [1.7, 1.1, 0.8, 0.5, 0.35], L, 0.04, 0.03)
    k = int(0.05 * SR)
    s[:k] += coloured(k, band(40, 700)) * np.exp(-t[:k] / 0.012) * 0.4      # soft felt sticks
    return s * vel * 0.6


def vibes(m, vel, L=4.5, motor=5.2):     # aluminium bars tuned 1:4:10; the motor's fans make it pulse
    s1, t = modes(hz(m), [1.0], [1.0], [3.0], L)
    s2, _ = modes(hz(m), [3.98, 9.96], [0.2, 0.05], [0.7, 0.22], L)
    trem = 1 - 0.32 * (0.5 + 0.5 * np.sin(TAU * motor * t + rng.uniform(0, TAU)))
    s = s1 * trem + s2
    k = int(0.006 * SR)
    s[:k] += coloured(k, band(700, 4500)) * np.exp(-t[:k] / 0.0015) * 0.12 * vel   # a yarn mallet
    return s * (1 - np.exp(-t / 0.001)) * vel * 0.45


def cym_roll(L, vel):                      # a suspended cymbal rolled with soft mallets, swelling
    t = np.arange(int(L * SR)) / SR
    sw = (t / L) ** 2.2
    s = coloured(len(t), band(900, 9000, 1)) * sw * (1 + 0.15 * np.sin(TAU * 13 * t))
    return s * vel * 0.2


def cym_bloom(vel, L=4.0):                 # ... and the plate left ringing after the last stroke
    t = np.arange(int(L * SR)) / SR
    s = coloured(len(t), band(700, 7000, 1)) * np.exp(-t / 1.4) + 0.5 * coloured(len(t), band(2500, 11000, 1)) * np.exp(-t / 0.8)
    return s * smooth(t / 0.02) * vel * 0.2


# --- harmony: D Lydian over double basses; the raised fourth (G#, or C# over G) is the colour of wonder
HARM = [  # start, end, double bass, low-string pad, piano roll (played upward)
    (0.00, 3.77, 38, [50, 57, 61, 66], [38, 45, 52, 54, 57, 61]),             # Dmaj9
    (3.77, 6.43, 38, [52, 56, 59, 64], [38, 45, 52, 56, 59, 64]),             # E/D: the Lydian II over the pedal
    (6.43, 9.15, 35, [50, 54, 57, 61], [35, 42, 50, 54, 57, 61]),             # Bm9
    (9.15, 11.80, 31, [47, 54, 59, 61], [31, 38, 47, 54, 59, 61]),            # Gmaj7#11
    (11.80, 13.10, 38, [50, 57, 61, 64, 66], [26, 38, 45, 52, 54, 57, 61, 64]),   # Dmaj9: the touch
    (13.10, 14.45, 38, [52, 56, 59, 64, 68], [38, 45, 52, 56, 59, 64]),       # E/D: the brightest lift
    (14.45, 15.80, 35, [50, 54, 57, 62], [35, 42, 50, 54, 57, 62]),           # Bm7
    (15.80, 17.17, 31, [47, 50, 54, 61], [31, 38, 47, 54, 61]),               # Gmaj7#11
    (17.17, 20.00, 38, [50, 54, 57, 59, 64], [26, 38, 45, 50, 54, 59, 64]),   # D6/9: resolved, open and bright
]

# --- melody: an original tune for the cellos, low and singing; it rises to the touch, then comes home
MELODY = [  # start, length, note
    (3.80, 1.00, 57), (4.80, 0.35, 59), (5.15, 0.40, 61), (5.55, 0.90, 64),
    (6.47, 0.70, 62), (7.17, 0.45, 61), (7.62, 0.45, 59), (8.07, 1.10, 54),
    (9.20, 0.60, 59), (9.80, 0.40, 61), (10.20, 0.40, 62), (10.60, 0.50, 64), (11.10, 0.70, 66),
    (11.80, 1.30, 69), (13.10, 0.55, 68), (13.65, 0.40, 66), (14.05, 0.42, 64),
    (14.47, 0.90, 62), (15.37, 0.43, 59), (15.80, 0.60, 61), (16.40, 0.80, 59),
    (17.20, 0.80, 57), (18.00, 0.50, 54), (18.50, 0.40, 52), (18.90, 1.80, 50),
]

# the little jelly's harp: one note a stroke while it climbs, quick figures once it flees
VISIT_NOTES = [(50,), (54,), (56,), (59,), (61,), (62,), (66,),
               (64, 66, 69), (64, 69), (68, 71), (64, 68), (71,), (69,), (71,)]
VISIT_VEL = [0.5, 0.52, 0.55, 0.57, 0.6, 0.6, 0.62, 0.72, 0.68, 0.64, 0.58, 0.5, 0.42, 0.34]
VIBE_NOTES = [69, 66, 68, 64, 59, 56]       # one bar per wave of light, sinking as they fade

PAD = curve([(0, 0.22), (3.5, 0.32), (9.0, 0.42), (11.0, 0.6), (11.8, 0.85), (13.5, 0.62), (17.0, 0.45), (20, 0.38)])
LEAD = curve([(3.5, 0.55), (9.0, 0.62), (11.8, 0.95), (14.5, 0.72), (18.0, 0.58), (20, 0.48)])
BASS = curve([(0, 0.35), (3.0, 0.45), (11.0, 0.6), (11.8, 0.85), (14.0, 0.58), (20, 0.5)])

# --- arrangement
PIANO, STR, CELLO, HARP, VIB, PERC = Bus(0.35), Bus(0.42), Bus(0.4), Bus(0.45), Bus(0.55), Bus(0.32)


def ppan(m):
    return np.clip((m - 50) / 30, -0.5, 0.5)


def chord_at(t):
    return next(h for h in HARM if h[0] <= t < h[1])


# low strings and double basses: a chord per harmony, each voice its own small section
for t0, t1, bass, pad, roll in HARM:
    L = t1 - t0 + 0.25
    STR.add(bowed(bass, t0, L, BASS, kind="cb", players=3, att=0.35, rel=1.0), t0, 0.05, 1.0)
    if t0 in (0.0, 11.8, 17.17):              # the deepest D under the opening, the touch and the close
        STR.add(bowed(bass - 12, t0, L, BASS, kind="cb", players=2, att=0.6, rel=1.2), t0, 0.0, 0.5)
    for i, m in enumerate(pad):
        kind = "va" if m >= 55 else "vc"
        STR.add(bowed(m, t0, L, PAD, kind=kind, players=4, att=0.55, rel=1.0), t0 + 0.02 * i,
                0.3 if kind == "va" else -0.3, 0.55)

# the cello tune, legato: each note slides in from the one before
prev = None
for t0, d, m in MELODY:
    glide = prev if prev is not None and abs(prev - m) <= 4 else None
    CELLO.add(bowed(m, t0, d + 0.06, LEAD, kind="vc", players=5, glide=glide, att=0.16, rel=0.6), t0, -0.12, 1.0)
    prev = m

# every bell pulse: the piano rolls its chord upward, as the ring of light runs from crown to rim;
# a double bass pizzicato underneath is the deep heartbeat
for b in BEATS[:-1]:
    tb = TOUCH if abs(b - 11.65) < 0.01 else b
    _, t1, bass, _, roll = chord_at(tb + 0.01)
    vel = 0.62 if tb == TOUCH else 0.42 + 0.03 * BEATS.index(b)
    for i, m in enumerate(roll):
        PIANO.add(piano(m, vel * (1 - 0.03 * i), min(t1 - tb + 0.4, 3.0)), tb + 0.085 * i + rng.normal(0, 0.004), ppan(m))
    PERC.add(harp(bass, 0.8, ring=0.55, dark=0.6), tb, 0.0, 0.9)

# the little jelly: a harp note on each stroke, panned to where it is on screen
for t0, notes, v in zip(VISIT, VISIT_NOTES, VISIT_VEL):
    x = np.interp(t0, *zip(*VISIT_X))
    pan = float(np.clip((x - 540) / 540 * 0.9, -0.8, 0.8))
    for k, m in enumerate(notes):
        HARP.add(harp(m, v * (1 - 0.06 * k)), t0 + 0.07 * k, pan)

# the touch: a mallet cymbal and a timpani roll swell into it ...
PERC.add(cym_roll(1.8, 0.9), TOUCH - 1.8, 0.25)
PERC.add(cym_bloom(0.55), TOUCH, 0.25)
tr = TOUCH - 1.2
while tr < TOUCH - 0.03:
    v = 0.05 + 0.35 * ((tr - (TOUCH - 1.2)) / 1.2) ** 2
    PERC.add(timpani(38, v, L=1.4), tr, -0.15)
    tr += rng.uniform(0.06, 0.075)
PERC.add(timpani(38, 0.95), TOUCH, -0.15)
PERC.add(timpani(33, 0.5), FLASH[2][0], -0.15)

# ... and the vibraphone answers every wave of light
for (tf, a), m in zip(FLASH, VIBE_NOTES):
    VIB.add(vibes(m, 0.25 + 0.6 * a), tf, 0.35 - 0.14 * VIBE_NOTES.index(m))


# --- hall: a concert-hall reverb - bass lingers, treble dies sooner, early reflections off the walls
def hall_ir(L=3.4):
    n = int(L * SR)
    t = np.arange(n) / SR
    f = np.fft.rfftfreq(n, 1 / SR)
    ir = np.zeros((2, n))
    for ch in (0, 1):
        W = np.fft.rfft(rng.normal(size=n))
        sig = sum(np.fft.irfft(W * b(f), n) * np.exp(-6.91 * t / rt)
                  for b, rt in ((band(1, 400), 2.8), (band(400, 3500), 2.2), (band(3500, 20000), 1.2)))
        sig *= smooth((t - 0.02) / 0.04)
        for _ in range(14):
            k = int(rng.uniform(0.008, 0.07) * SR)
            sig[k] += rng.uniform(0.3, 1.0) * (1 - k / (0.08 * SR)) * 4 * (np.abs(sig).max())
        ir[ch] = sig / np.sqrt((sig ** 2).sum())
    return ir


def convolve(x, h):
    m = 1 << (len(x) + len(h) - 1).bit_length()
    return np.fft.irfft(np.fft.rfft(x, m) * np.fft.rfft(h, m), m)[:len(x)]


# --- mix: the buses, the hall, a gentle glue compressor and a soft limiter
def compress(x, thr=-18.0, ratio=2.0, att=0.015, rel=0.3, blk=256):
    nb = x.shape[1] // blk
    lev = 20 * np.log10(np.sqrt((x[:, :nb * blk].reshape(2, nb, blk) ** 2).mean((0, 2))) + 1e-9)
    want = np.minimum(0, (thr - lev) * (1 - 1 / ratio))
    g, out = 0.0, np.zeros(nb)
    a_a, a_r = math.exp(-blk / SR / att), math.exp(-blk / SR / rel)
    for i in range(nb):
        k = a_a if want[i] < g else a_r
        g = k * g + (1 - k) * want[i]
        out[i] = g
    gain = 10 ** (np.interp(np.arange(x.shape[1]), np.arange(nb) * blk + blk / 2, out) / 20)
    return x * gain


BUSES = {"piano": (PIANO, 0.62), "strings": (STR, 0.8), "cello": (CELLO, 1.2), "harp": (HARP, 0.5),
         "vibes": (VIB, 1.0), "perc": (PERC, 0.6)}
LOUD = -15.5                               # RMS target in dB; about -13.7 LUFS once measured


def mix():
    dry = sum(B.x * g for B, g in BUSES.values())
    send = sum(B.x * g * B.send for B, g in BUSES.values())
    ir = hall_ir()
    wet = np.stack([convolve(send[0] * 0.75 + send[1] * 0.25, ir[0]), convolve(send[1] * 0.75 + send[0] * 0.25, ir[1])])
    x = dry + 0.55 * wet
    lo = band(30, 20000, 1)                    # no rumble below the double basses' lowest D
    m = 1 << (x.shape[1] - 1).bit_length()
    x = np.stack([np.fft.irfft(np.fft.rfft(c, m) * lo(np.fft.rfftfreq(m, 1 / SR)), m)[:x.shape[1]] for c in x])
    x = compress(x)
    x = x[:, :int(DUR * SR)]
    t = np.arange(x.shape[1]) / SR
    x *= smooth(t / 0.01) * (1 - smooth((t - (DUR - 1.1)) / 1.1))       # let the last chord fade away
    x *= 10 ** (LOUD / 20) / np.sqrt((x ** 2).mean())
    x = 0.97 * np.tanh(x / 0.97)
    return x


def write_wav(path, x):
    y = (np.clip(x, -1, 1) * 32767).astype("<i2").T.copy()
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(y.tobytes())


if __name__ == "__main__":
    out = mix()
    name = os.path.splitext(os.path.basename(__file__))[0] + ".wav"
    write_wav(name, out)
    print(f"wrote {name}  peak {float(np.abs(out).max()):.3f}  rms {20 * np.log10(np.sqrt((out ** 2).mean())):.1f} dB")
    if "--stats" in sys.argv:                  # per-bus level, and where each bus sits in the spectrum
        for k, (B, g) in BUSES.items():
            x = B.x[:, :int(DUR * SR)] * g
            X = np.abs(np.fft.rfft(x.mean(0))) ** 2
            f = np.fft.rfftfreq(x.shape[1], 1 / SR)
            cen = float((X * f).sum() / X.sum())
            lowshare = float(X[f < 150].sum() / X.sum())
            print(f"  {k:8s} rms {20 * np.log10(np.sqrt((x ** 2).mean()) + 1e-12):6.1f} dB   centroid {cen:6.0f} Hz   below 150 Hz {lowshare:4.0%}")
    if "--mux" in sys.argv:
        k = sys.argv.index("--mux")
        ff = shutil.which("ffmpeg") or "ffmpeg"
        subprocess.run([ff, "-y", "-loglevel", "error", "-i", sys.argv[k + 1], "-i", name,
                        "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
                        "-shortest", "-movflags", "+faststart", sys.argv[k + 2]], check=True)
        print("muxed ->", sys.argv[k + 2])
