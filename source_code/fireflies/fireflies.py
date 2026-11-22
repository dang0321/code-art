"""Draw fireflies at three depths in a quiet dusk forest."""

import argparse
import inspect
import math
from pathlib import Path
import re
import shutil
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


WIDTH, HEIGHT = 1080, 1920
FPS, DURATION = 30, 20
CELL_W, CELL_H = 8, 10
COLS, ROWS = 135, 137
ART_HEIGHT = ROWS * CELL_H
GLYPHS = " .,:-;~+*ox|/\\"
SYMBOL = {glyph: index for index, glyph in enumerate(GLYPHS)}
X, Y = np.meshgrid(
    (np.arange(COLS, dtype=np.float32) + 0.5) * CELL_W,
    (np.arange(ROWS, dtype=np.float32) + 0.5) * CELL_H,
)


def clamp(value):
    return np.clip(value, 0, 1)


def smoothstep(value):
    value = clamp(value)
    return value * value * (3 - 2 * value)


def hash_grid(x, y):
    x = np.asarray(x, dtype=np.uint32)
    y = np.asarray(y, dtype=np.uint32)
    value = (x * np.uint32(1597334677)) ^ (y * np.uint32(3812015801))
    value ^= value >> np.uint32(16)
    value = np.multiply(value, np.uint32(2246822519), dtype=np.uint32)
    value ^= value >> np.uint32(13)
    return (value & np.uint32(65535)).astype(np.float32) / 65535


def noise(x, y):
    """Interpolate neighboring values so the flame can flow between frames."""
    ix, iy = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32)
    dx, dy = smoothstep(x - ix), smoothstep(y - iy)
    low = hash_grid(ix, iy) * (1 - dx) + hash_grid(ix + 1, iy) * dx
    high = hash_grid(ix, iy + 1) * (1 - dx) + hash_grid(ix + 1, iy + 1) * dx
    return low * (1 - dy) + high * dy


def grid_point(x, y):
    return round(x / CELL_W - 0.5), round(y / CELL_H - 0.5)


def stand_of_trees(rng, points, solidity):
    """Spread tapered trunks and irregular branches around the open glade."""
    layer = np.zeros((ROWS, COLS), dtype=np.uint8)
    for x, foot, radius, lean in points:
        high = rng.uniform(-110, 81)
        bark = ((x - radius, foot + 14), (x - radius * 0.45 + lean * 0.48, (foot + high) * 0.5),
                (x - radius * 0.28 + lean, high), (x + radius * 0.40 + lean, high),
                (x + radius * 0.55 + lean * 0.44, (foot + high) * 0.51),
                (x + radius * 0.77, foot + 12))
        hull = np.asarray([grid_point(*point) for point in bark], dtype=np.int32)
        cv2.fillPoly(layer, [hull], solidity, lineType=cv2.LINE_AA)
        for side in (-1, 1):
            root_y = foot * rng.uniform(0.20, 0.55)
            anchor_x = x + lean * 0.52
            length = rng.uniform(56, 179) * (0.72 + radius / 55)
            joint = (anchor_x + side * length * 0.44, root_y - length * 0.36)
            tip = (anchor_x + side * length, root_y - length * 0.62)
            cv2.line(layer, grid_point(anchor_x, root_y), grid_point(*joint),
                     solidity, max(1, round(radius / 17)), cv2.LINE_AA)
            cv2.line(layer, grid_point(*joint), grid_point(*tip), solidity, 1, cv2.LINE_AA)
            twig = (tip[0] - side * 9, tip[1] - 73)
            cv2.line(layer, grid_point(*joint), grid_point(*twig), solidity, 1, cv2.LINE_AA)
    return layer.astype(np.float32) / 255


def fern_fronds():
    """Use long arching stems and paired leaflets in the foreground."""
    layer = np.zeros((ROWS, COLS), dtype=np.uint8)
    for base_x, base_y, fan, height in ((91, 1280, 1, 193), (232, 1380, -1, 179),
                                        (946, 1321, -1, 238), (1031, 1258, 1, 214)):
        for tilt in (-0.75, -0.41, -0.06, 0.31, 0.74):
            direction = 0.20 * fan + tilt
            length = height * (1 - abs(direction) * 0.27)
            u = np.linspace(0, 1, 43)
            xx = base_x + length * (direction * u + 0.33 * fan * u ** 2)
            yy = base_y - length * (1.24 * u - 0.39 * u ** 2)
            center = np.asarray([grid_point(x, y) for x, y in zip(xx, yy)], dtype=np.int32)
            cv2.polylines(layer, [center], False, 180, 1, cv2.LINE_AA)
            for i in range(7, 41, 3):
                spread = (0.12 + 0.095 * (1 - u[i])) * length * (1 - u[i]) ** 0.7
                for side in (-1, 1):
                    tip = (xx[i] + side * spread, yy[i] - spread * 0.71)
                    cv2.line(layer, tuple(center[i]), grid_point(*tip), 156, 1, cv2.LINE_AA)
    return layer.astype(np.float32) / 255


def make_glade():
    """Set dark trees against a small opening with the last light of dusk."""
    rng = np.random.default_rng(149)
    grain = noise(X * 0.101 + 8, Y * 0.084 + 44)
    coarse = noise(X * 0.014 + 77, Y * 0.024 + 2)
    fibers = noise(X * 0.072 + 111, Y * 0.017)
    open_air = np.exp(-((X - 543) / 404) ** 2 - ((Y - 636) / 415) ** 2)
    above = np.exp(-((X - 635) / 720) ** 2 - ((Y - 263) / 628) ** 2)
    last_light = np.exp(-((X - 466) / 287) ** 2 - ((Y - 692) / 130) ** 2)
    haze = smoothstep((Y - 608) / 218) * smoothstep((1064 - Y) / 181) * open_air
    base = np.full((*X.shape, 3), (4.0, 8.0, 14.0), dtype=np.float32)
    base += above[..., None] * np.float32((8, 18, 28))
    base += open_air[..., None] * np.float32((12, 22, 23))
    base += last_light[..., None] * np.float32((36, 24, 5))
    base += (haze * (0.29 + 0.71 * coarse))[..., None] * np.float32((14, 19, 12))

    far_specs = [(x, rng.uniform(725, 818), rng.uniform(5, 16), rng.uniform(-31, 27))
                 for x in np.linspace(-15, WIDTH + 16, 24) + rng.uniform(-19, 25, 24)]
    far = stand_of_trees(rng, far_specs, 128)
    mid = stand_of_trees(rng, ((102, 919, 27, 58), (207, 872, 21, -15),
                                (293, 839, 12, -33), (760, 849, 16, 22),
                                (845, 917, 30, 12), (1040, 1055, 28, -46)), 185)
    near = stand_of_trees(rng, ((-19, 1109, 55, -25), (139, 1207, 49, -114),
                                 (974, 1211, 51, 74), (1094, 1369, 57, 24)), 242)
    crown = smoothstep((coarse + 0.17 * np.sin(X * 0.017) - 0.43) / 0.40)
    crown *= smoothstep((560 - Y + 101 * (1 - open_air)) / 474)
    foliage = crown * (0.60 + 0.40 * grain)
    far *= 0.85
    silhouette = 1 - (1 - near) * (1 - mid) * (1 - far)
    base *= (1 - 0.84 * silhouette[..., None])
    base *= (1 - 0.55 * foliage[..., None])

    floor = smoothstep((Y - 772 + 21 * np.sin(X * 0.018)) / 189)
    path_width = 44 + np.maximum(Y - 780, 0) * 0.44
    trail = np.exp(-((X - 552 - 24 * np.sin(Y * 0.011)) / path_width) ** 2)
    grass = floor * (1 - 0.84 * trail)
    floor_tone = (0.47 * grass + 0.15 * floor) * (0.49 + 0.51 * coarse)
    base += floor_tone[..., None] * np.float32((11, 20, 18))
    fronds = fern_fronds()
    marks, rgb, base = landscape_glyphs(base, silhouette, floor, grass, fronds,
                                        grain, coarse, fibers, foliage, haze, last_light)
    near_occlusion = np.maximum(near, fronds)
    return marks, rgb, base, far, mid, near_occlusion


def landscape_glyphs(base, silhouette, floor, grass, fronds, grain, coarse, fibers, foliage, haze, last_light):
    """Use visible character values for the clearing, bark and ground cover."""
    marks = np.zeros(X.shape, dtype=np.intp)
    rgb = np.zeros((*X.shape, 3), dtype=np.float32)

    air = (1 - floor * 0.86) * (1 - silhouette * 0.60)
    dither = 0.59 * grain + 0.41 * coarse
    opening = base[..., 1] / 48
    tone = air * (0.52 * opening + 0.21 * haze + 0.20 * dither)
    marks = np.select([tone > 0.52, tone > 0.36, tone > 0.11],
                      [SYMBOL["-"], SYMBOL[":"], SYMBOL["."]], default=0)
    sky_ink = tone * (0.70 + 0.30 * grain)
    rgb += sky_ink[..., None] * (np.float32((61, 121, 153)) + last_light[..., None] * np.float32((66, 28, -12)))

    canopy = (foliage > 0.24) & (silhouette < 0.38) & (dither > 0.30)
    leaf = np.select([grain > 0.71, fibers > 0.56], [SYMBOL["x"], SYMBOL[":"]], default=SYMBOL["."])
    marks[canopy] = leaf[canopy]
    rgb[canopy] = (0.22 + 0.32 * coarse[canopy, None]) * np.float32((31, 57, 74))

    bark = (silhouette > 0.29) & (floor < 0.79)
    stripes = 0.52 * fibers + 0.28 * grain + 0.20 * coarse
    edges = np.maximum(0, silhouette - cv2.erode(silhouette, np.ones((3, 3), dtype=np.uint8)))
    bark_values = (0.29 + 0.56 * stripes) * (0.56 + 0.44 * edges)
    marks[bark] = np.select([stripes[bark] > 0.55, edges[bark] > 0.32],
                            [SYMBOL["|"], SYMBOL["/"]], default=SYMBOL[":"])
    rgb[bark] = bark_values[bark, None] * np.float32((37, 64, 68))

    brush = grass * (0.39 + 0.61 * dither) * (0.40 + 0.60 * fibers)
    plant = brush > 0.16
    plant_ink = (0.28 + 0.77 * brush) * (0.75 + 0.25 * grain)
    marks[plant] = np.select([grain[plant] > 0.70, fibers[plant] > 0.61, grain[plant] < 0.29],
                             [SYMBOL["/"], SYMBOL[":"], SYMBOL[","]], default=SYMBOL["."])
    rgb[plant] = plant_ink[plant, None] * np.float32((44, 99, 66))

    trail = (floor > 0.41) & (brush <= 0.16) & (dither > 0.26)
    marks[trail] = np.where(dither[trail] > 0.62, SYMBOL["~"], SYMBOL["."])
    rgb[trail] = (0.17 + 0.45 * dither[trail, None]) * np.float32((34, 72, 80))

    tips = (fronds > 0.17) & ((grain > 0.33) | (fronds > 0.50))
    marks[tips] = np.where(grain[tips] > 0.47, SYMBOL["/"], SYMBOL[chr(92)])
    rgb[tips] = (0.27 + 0.46 * fronds[tips, None]) * np.float32((39, 91, 55))
    shadow = np.float32((3.0, 6.0, 10.0)) + base * 0.22
    shadow *= 1 - (0.27 * fronds)[..., None]
    return marks, rgb, shadow


class Fireflies:
    def __init__(self):
        rng = np.random.default_rng(18073)
        count = 113
        depth = rng.choice(3, count, p=(0.44, 0.41, 0.15))
        x = rng.uniform(14, 1066, count)
        floor = np.choose(depth, (882, 1035, 1257))
        y = floor - rng.uniform(28, 424, count)
        self.depth = depth
        self.x, self.y = x.astype(np.float32), y.astype(np.float32)
        self.period = rng.uniform(2.2, 5.9, count).astype(np.float32)
        self.phase = rng.uniform(0, math.tau, count).astype(np.float32)
        self.bias = rng.uniform(0.35, 1.0, count).astype(np.float32)
        self.tint = rng.uniform(0.0, 1.0, count).astype(np.float32)
        self.drift = rng.uniform(-1, 1, count).astype(np.float32)
        self.first = rng.uniform(-4.0, 3.6, count).astype(np.float32)

    def flight(self, time):
        """Move distant lights gently as closer ones cover longer paths."""
        reach = np.array((22, 53, 95), dtype=np.float32)[self.depth]
        phase = self.phase
        drift = np.sin(0.37 * time + phase) + 0.29 * np.sin(0.74 * time + phase * 2.7)
        curl = 0.74 * np.sin(time * 0.67 + phase * 1.5) + 0.24 * np.sin(1.20 * time + phase)
        x = self.x + reach * (drift + 0.14 * self.drift * np.sin(time * 0.23))
        y = self.y + reach * curl * 0.57
        return x, y

    def pulses(self, time):
        """Let each flash swell and fade on its own timing."""
        phase = (time / self.period + self.phase / math.tau) % 1
        onset = 0.22 + 0.085 * self.drift
        distance = (phase - onset + 0.5) % 1.0 - 0.5
        flash = np.exp(-(distance / (0.112 + 0.057 * self.bias)) ** 2)
        echo = 0.16 * np.exp(-((phase - onset - 0.22) / 0.100) ** 2)
        awake = smoothstep((time - self.first) / 2.30)
        dim = np.array((0.10, 0.15, 0.22), dtype=np.float32)[self.depth]
        return (dim + (0.55 + self.bias * 0.54) * flash + echo) * awake

    def passes(self, time):
        """A few close passes carry the strongest light through the scene."""
        course = np.asarray((
            (-1.4, 5.5, 538, 332, 864, 769, 36, 0.72),
            (2.0, 8.3, 117, 664, 1024, 726, 36, 0.39),
            (6.4, 9.4, 1028, 347, 849, 1009, 62, 0.84),
            (10.9, 8.6, 250, 901, 1105, 796, 51, 0.51),
            (16.8, 6.1, 878, 553, 744, 851, 41, 0.70),
        ), dtype=np.float32)
        start, duration, x0, x1, y0, y1, arc, tint = course.T
        progress = (time - start) / duration
        u = np.clip(progress, 0, 1)
        x = x0 + (x1 - x0) * smoothstep(u) + 23 * np.sin(u * math.tau) * np.sin(np.pi * u)
        y = y0 + (y1 - y0) * u - arc * np.sin(np.pi * u) + 22 * np.sin(time * 0.98)
        life = smoothstep(u / 0.16) * smoothstep((1 - u) / 0.19)
        flash = life * (0.49 + 0.51 * (0.5 + 0.5 * np.sin(time * 1.46 + tint * 7)) ** 1.7)
        return x, y, flash, tint

    def draw(self, marks, rgb, base, time, occlusion):
        x, y = self.flight(time)
        close_x, close_y, close_flash, close_tint = self.passes(time)
        x, y = np.r_[x, close_x], np.r_[y, close_y]
        glow = np.r_[self.pulses(time), close_flash]
        depth = np.r_[self.depth, np.full(close_flash.shape, 3, dtype=np.intp)]
        warmth = np.r_[self.tint, close_tint]
        fx, fy = x / CELL_W - 0.5, y / CELL_H - 0.5
        cx, cy = np.floor(fx).astype(np.intp), np.floor(fy).astype(np.intp)
        center_x, center_y = np.clip(np.rint(fx).astype(np.intp), 0, COLS - 1), np.clip(np.rint(fy).astype(np.intp), 0, ROWS - 1)
        far, mid, near = (mask[center_y, center_x] for mask in occlusion)
        blocked = np.where(depth == 0, np.maximum(near, np.maximum(mid, far * 0.8)),
                           np.where(depth == 1, near, near * 0.28))
        glow *= 1 - blocked * 0.88
        colors = np.float32((208, 255, 115)) * (1 - warmth[:, None])
        colors += np.float32((255, 235, 120)) * warmth[:, None]

        # Blend the soft light across cells as the bright glyph changes position.
        weights = np.zeros((ROWS, COLS), dtype=np.float32)
        hot = np.zeros((ROWS, COLS, 3), dtype=np.float32)
        halos = np.zeros((ROWS, COLS, 3), dtype=np.float32)
        near_halos = np.zeros_like(halos)
        core_scale = np.float32((0.83, 1.39, 1.83, 2.11))[depth]
        for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
            col, row = cx + dx, cy + dy
            value = glow * np.maximum(0, 1 - np.abs(fx - col)) * np.maximum(0, 1 - np.abs(fy - row))
            okay = (row >= 0) & (row < ROWS) & (col >= 0) & (col < COLS) & (value > 0.002)
            where = row[okay], col[okay]
            hue = colors[okay]
            np.add.at(halos, where, hue * (value[okay] * core_scale[okay] ** 1.18)[:, None])
            np.add.at(near_halos, where, hue * (value[okay] * (depth[okay] >= 2))[:, None])

        glyph_x, glyph_y = np.rint(fx).astype(np.intp), np.rint(fy).astype(np.intp)
        on_grid = (glyph_x >= 0) & (glyph_x < COLS) & (glyph_y >= 0) & (glyph_y < ROWS)
        cores = glyph_y[on_grid], glyph_x[on_grid]
        strength = glow[on_grid] * core_scale[on_grid] * 0.70
        np.maximum.at(weights, cores, strength)
        np.maximum.at(hot, cores, colors[on_grid] * (strength ** 0.64)[:, None])
        points = weights > 0.035
        marks[points] = np.select([weights[points] > 0.37, weights[points] > 0.145],
                                  [SYMBOL["*"], SYMBOL["+"]], default=SYMBOL["."])
        rgb[points] = np.maximum(rgb[points], hot[points])
        base += cv2.GaussianBlur(halos, (0, 0), 0.75) * 0.81
        base += cv2.GaussianBlur(near_halos, (0, 0), 1.81) * 0.92
        self.close_glow(marks, rgb, base, x, y, glow, depth, colors)
        visible = (x > 0) & (x < WIDTH) & (y > 0) & (y < ART_HEIGHT)
        return int(np.count_nonzero(visible & (glow > 0.38)))

    def close_glow(self, marks, rgb, base, x, y, glow, depth, colors):
        """Keep the nearest sparkles bright with a narrow center and soft edge."""
        close = (depth == 3) & (glow > 0.008)
        for px, py, light, color in zip(x[close], y[close], glow[close], colors[close]):
            distance = (X - px) ** 2 + (Y - py) ** 2
            bloom = 0.22 * np.exp(-distance / 96)
            bloom += 0.14 * np.exp(-distance / 427)
            bloom += 0.055 * np.exp(-distance / 1241)
            base += (light * bloom)[..., None] * color
            col, row = grid_point(px, py)
            if 0 <= col < COLS and 0 <= row < ROWS and light > 0.14:
                marks[row, col] = SYMBOL["*"]
                rgb[row, col] = np.maximum(rgb[row, col], color * (0.73 + 0.75 * light) + np.float32((19, 13, 27)))


class Twilight:
    def __init__(self):
        self.marks, self.rgb, self.base, *self.occlusion = make_glade()
        self.flies = Fireflies()
        self.atlas = make_glyph_atlas()
        self.panel = CodePanel()
        self.font = mono(23)

    def render(self, time):
        marks, rgb, glow = self.marks.copy(), self.rgb.copy(), self.base.copy()
        active = self.flies.draw(marks, rgb, glow, time, self.occlusion)
        light = cv2.resize(glow, (WIDTH, ART_HEIGHT), interpolation=cv2.INTER_LINEAR)
        ink = self.atlas[marks][..., None] * rgb[:, :, None, None, :]
        ink = ink.transpose(0, 2, 1, 3, 4).reshape(ART_HEIGHT, WIDTH, 3)
        frame = Image.new("RGB", (WIDTH, HEIGHT), (4, 8, 14))
        frame.paste(Image.fromarray(np.uint8(np.clip(light + ink, 0, 255))), (0, 0))
        draw = ImageDraw.Draw(frame)
        draw.text((33, 30), "fireflies.py", font=self.font, fill="#dae4d8")
        draw.text((33, 60), f"lit     {active:02d}", font=self.font, fill="#a1b4a8")
        draw.text((WIDTH - 34, 30), f"t {time:05.2f}s", font=self.font, fill="#a1b4a8", anchor="ra")
        self.panel.draw(frame, time)
        return frame


def mono(size):
    for name in ("consola.ttf", "DejaVuSansMono.ttf", "LiberationMono-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size)


def make_glyph_atlas():
    font = mono(13)
    bright = mono(17)
    atlas = []
    for character in GLYPHS:
        mask = Image.new("L", (CELL_W, CELL_H))
        draw = ImageDraw.Draw(mask)
        if character == "*":
            bounds = bright.getbbox(character, anchor="mt")
            height = bounds[3] - bounds[1]
            draw.text((CELL_W / 2, (CELL_H - height) / 2), character, font=bright, anchor="mt", fill=255)
        else:
            draw.text((CELL_W / 2, 9), character, font=font, anchor="ms", fill=255)
        atlas.append(np.asarray(mask, dtype=np.float32) / 255)
    return np.stack(atlas)


SYNTAX = re.compile(r'(#.+$)|(\b(?:def|for|in|if|return|import|as|and|or|else|from)\b)|'
                    r'(\b\d+(?:\.\d+)?\b)|("[^"\n]*")|'
                    r'(\b[a-zA-Z_]\w*(?=\())|([=+*\-/<>])')
TOKEN_COLORS = ("#8992a5", "#c792e8", "#efaa75", "#a6c484", "#91b9de", "#91a7be")
LINE_HEIGHT = 29
PANEL_W, PANEL_H = 1020, 526
HEADER_H = 47


class CodePanel:
    def __init__(self):
        self.font = mono(24)
        self.advance = self.font.getlength("0")
        source = Path(__file__).read_text(encoding="utf-8").splitlines()
        self.sections = []
        for function in (landscape_glyphs, Fireflies.flight, Fireflies.pulses, Fireflies.draw):
            lines, line_number = inspect.getsourcelines(function)
            start = max(0, line_number - 4)
            end = start + len(lines) + 15
            body = Image.new("RGB", (PANEL_W, (end - start) * LINE_HEIGHT), "#10131a")
            draw = ImageDraw.Draw(body)
            for index, line in enumerate(source[start:end]):
                y = index * LINE_HEIGHT + 5
                draw.text((51, y), str(start + index + 1), font=self.font, fill="#758094", anchor="rt")
                self.draw_line(draw, line[:66], y)
            self.sections.append((body, len(lines)))
        self.mask = Image.new("L", (PANEL_W, PANEL_H))
        ImageDraw.Draw(self.mask).rounded_rectangle((0, 0, PANEL_W - 1, PANEL_H - 1), radius=17, fill=255)

    def draw_line(self, draw, content, y):
        cursor = 0
        for token in SYNTAX.finditer(content):
            self.draw_span(draw, content[cursor:token.start()], cursor, y, "#c3c9d5")
            self.draw_span(draw, token.group(), token.start(), y, TOKEN_COLORS[token.lastindex - 1])
            cursor = token.end()
        self.draw_span(draw, content[cursor:], cursor, y, "#c3c9d5")

    def draw_span(self, draw, text, column, y, color):
        draw.text((70 + column * self.advance, y), text, font=self.font, fill=color, anchor="lt")

    def draw(self, frame, time):
        panel = Image.new("RGB", (PANEL_W, PANEL_H), "#10131a")
        part = time / (DURATION / len(self.sections))
        section = min(len(self.sections) - 1, int(part))
        lines, count = self.sections[section]
        scroll = min(20, max(2, count - 7)) * (part - section)
        top = round(scroll * LINE_HEIGHT)
        body = lines.crop((0, top, PANEL_W, top + PANEL_H - HEADER_H - 3))
        highlight = round((int(scroll) + 6 - scroll) * LINE_HEIGHT)
        ribbon = body.crop((0, highlight, PANEL_W, highlight + LINE_HEIGHT))
        tint = Image.new("RGB", ribbon.size, "#3c4a69")
        body.paste(Image.blend(ribbon, tint, 0.23), (0, highlight))
        panel.paste(body, (0, HEADER_H))

        draw = ImageDraw.Draw(panel)
        draw.line((0, HEADER_H - 1, PANEL_W, HEADER_H - 1), fill="#292e3b", width=2)
        for index, color in enumerate(("#ff665c", "#ffbe4c", "#38c66d")):
            left = 18 + index * 24
            draw.ellipse((left, 17, left + 13, 30), fill=color)
        draw.text((91, 23), "fireflies.py", font=self.font, fill="#d6dbe3", anchor="lm")
        draw.text((1001, 23), "running   30 fps", font=self.font, fill="#a4adbe", anchor="rm")
        draw.ellipse((763, 18, 774, 29), fill="#52ce73")
        draw.rounded_rectangle((0, 0, PANEL_W - 1, PANEL_H - 1), radius=17, outline="#333b4b", width=2)
        frame.paste(panel, (30, 1364), self.mask)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--frame", type=int, help="Preview one frame as a PNG.")
    args = parser.parse_args()
    cv2.setNumThreads(2)
    scene = Twilight()
    if args.frame is not None:
        if not 0 <= args.frame < FPS * DURATION:
            parser.error("--frame must be between 0 and 599.")
        scene.render(args.frame / FPS).save(args.output)
        return
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        parser.error("FFmpeg was not found on PATH.")
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-n", "-f", "rawvideo",
               "-pixel_format", "rgb24", "-video_size", f"{WIDTH}x{HEIGHT}", "-framerate", str(FPS),
               "-i", "-", "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "16",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output)]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
        try:
            for index in range(FPS * DURATION):
                encoder.stdin.write(scene.render(index / FPS).tobytes())
                if (index + 1) % 150 == 0:
                    print(f"{(index + 1) // FPS:02d}/{DURATION}s rendered", flush=True)
        finally:
            encoder.stdin.close()
        if encoder.wait() != 0:
            raise SystemExit("FFmpeg could not encode the video.")


if __name__ == "__main__":
    main()
