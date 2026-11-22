"""Render the birth, branching, and afterglow of filaments in an ASCII plasma globe."""
import argparse
import inspect
import math
from pathlib import Path
import re
import shutil
import subprocess
import textwrap

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1080, 1920
ART_HEIGHT = 1344
CELL_W, CELL_H = 9, 12
COLS, ROWS = WIDTH // CELL_W, ART_HEIGHT // CELL_H
SPINE_SCALE = 3
SPINE_WIDTH, SPINE_HEIGHT = WIDTH // SPINE_SCALE, ART_HEIGHT // SPINE_SCALE
FPS, DURATION = 30, 20
CENTER = np.float32((520, 640))
RADIUS = 389
TAU = math.tau
GLYPHS = " .:+*|-/\\"
MARK = {character: index for index, character in enumerate(GLYPHS)}

CY, CX = np.indices((ROWS, COLS), dtype=np.float32)
CX = (CX + .5) * CELL_W
CY = (CY + .5) * CELL_H
DX, DY = CX - CENTER[0], CY - CENTER[1]
DISTANCE = np.hypot(DX, DY)
ANGLE = np.arctan2(DY, DX)
S = np.linspace(0, 1, 205, dtype=np.float32)
GLASS = np.linspace(0, 1, 45, dtype=np.float32)
GLASS_GRAIN = np.random.default_rng(41).random(len(GLASS), dtype=np.float32)
RADIAL = 48 + (RADIUS - 51) * S
TAPER = np.maximum(0, np.sin(np.pi * S)) ** .71

COLD = np.float32((.39, .52, 1.0))
ROSE = np.float32((1.0, .29, .68))
PALE_ROSE = np.float32((1.0, .62, .85))
WHITE = np.float32((.83, .89, 1.0))


def smoothstep(start, end, value):
    scaled = np.clip((value - start) / (end - start), 0, 1)
    return scaled * scaled * (3 - 2 * scaled)


def ellipse_glow(x, y, horizontal, vertical):
    return np.exp(-((x / horizontal) ** 2 + (y / vertical) ** 2))


def tangent_glyphs(dx, dy):
    horizontal = np.abs(dx)
    vertical = np.abs(dy)
    return np.select(
        (horizontal > 1.9 * vertical, vertical > 2.2 * horizontal, dx * dy > 0),
        (MARK["-"], MARK["|"], MARK["\\"]),
        default=MARK["/"],
    )


def soft_noise(rng, count):
    position = S * (count - 1)
    left = np.minimum(position.astype(np.intp), count - 2)
    fraction = position - left
    blend = fraction * fraction * (3 - 2 * fraction)
    knots = rng.uniform(-1, 1, count).astype(np.float32)
    return knots[left] + (knots[left + 1] - knots[left]) * blend


def project(radius, azimuth, elevation):
    # Filaments on the front and back can end well inside the glass outline.
    distance = radius * np.cos(elevation)
    return CENTER + np.stack((distance * np.cos(azimuth), distance * np.sin(azimuth)), axis=-1)


def filament_light(s, age, rise, stop, phase, strength):
    head = min(1.05, 1.05 * max(0, age / rise) ** .78)
    emerged = smoothstep(-.04, .035, head - s)
    # Let the old channel break unevenly as its light dies.
    cutoff = stop + .041 * np.sin(18 * s + phase)
    live = 1 - smoothstep(-.020, .11, age - cutoff)
    packet = np.exp(-((s - (age % .27) / .27) / .105) ** 2)
    flicker = .89 + .11 * np.sin(12 * s - 25 * age + phase)
    front = np.exp(-((s - head + .024) / .055) ** 2)
    front *= 1 - smoothstep(rise * .67, rise * 1.28, age)
    crackle = .85 + .15 * smoothstep(-.5, .93, np.sin(43 * age - 16 * s + phase))
    return emerged * live * strength * crackle * (.75 * flicker + .27 * packet + .54 * front)


def light_on_glass(field, angle, elevation, sweep, energy, age):
    # The pink fan follows the moving attachment on the inside of the sphere.
    width = .055 + .074 * energy
    turn = sweep * width * GLASS
    radii = RADIUS - 3 - 15 * GLASS ** 1.6
    skirt = project(radii, angle - turn, elevation + .095 * GLASS)
    tint = np.broadcast_to(ROSE, (len(GLASS), 3))
    field.path(skirt, energy * (1 - GLASS) ** .85 * .95, tint, GLASS_GRAIN)

    curl = angle + sweep * width * GLASS * .41
    tongue = project(radii - GLASS * 17, curl, elevation - .075 * GLASS)
    light = energy * (.79 + .21 * np.sin(48 * age + GLASS * 8))
    field.path(tongue, light * (1 - GLASS) ** 1.6, tint, GLASS_GRAIN[::-1])

    tip = project(RADIUS - 5, angle, elevation)
    x, y = CX - tip[0], CY - tip[1]
    normal = x * np.cos(angle) + y * np.sin(angle)
    tangent = -x * np.sin(angle) + y * np.cos(angle)
    flare = ellipse_glow(normal, tangent + 8 * sweep, 12, 27)
    field.region(flare * energy * .98, ROSE, MARK[":"])
    field.region(ellipse_glow(x, y, 6.4, 6.8) * energy * 1.14, PALE_ROSE, MARK["+"])


class TravelingBends:
    def __init__(self, rng, lifetime, count):
        self.birth = np.sort(rng.uniform(-.31, lifetime + .03, count))
        self.origin = rng.uniform(.07, .64, count)
        self.speed = rng.uniform(.75, 2.25, count)
        self.duration = rng.uniform(.18, .52, count)
        self.width = rng.uniform(.10, .25, count)
        self.skew = rng.uniform(.52, 1.58, count)
        self.turn = rng.uniform(24, 84, count) * rng.choice((-1, 1), count)
        self.depth = rng.uniform(-.19, .19, count)

    def sample(self, distance, time):
        age = (time - self.birth) / self.duration
        strength = smoothstep(0, .23, age) * (1 - smoothstep(.52, 1, age))
        focus = self.origin + self.speed * (time - self.birth)
        gap = distance[None, :] - focus[:, None]
        width = self.width[:, None] * np.where(gap < 0, self.skew[:, None], 1)
        bend = np.maximum(0, 1 - (gap / width) ** 2) ** 2
        bend *= strength[:, None]
        return self.turn @ bend, self.depth @ bend


class Discharge:
    def __init__(self, rng, born, angle):
        self.born = born
        self.rise = rng.uniform(.052, .115)
        self.stop = rng.uniform(.26, .98)
        self.strength = rng.uniform(.70, 1.20)
        self.angle = angle + rng.normal(0, .32)
        self.elevation = np.clip(rng.normal(0, .49), -.98, .98)
        self.phase = rng.uniform(0, TAU)
        self.spin = rng.choice((-1, 1)) * rng.uniform(.09, .44)
        self.curve = 19 * soft_noise(rng, 5) + 6 * soft_noise(rng, 18)
        self.depth = .14 * soft_noise(rng, 5)
        self.grain = rng.random(len(S), dtype=np.float32)
        self.mixes = smoothstep(.62, 1.01, S)
        self.split = rng.random()
        self.switch_route = self.split > .50
        self.branch_start = rng.integers(43, 125)
        self.branch_angle = rng.choice((-1, 1)) * rng.uniform(.14, .42)
        self.branch_depth = rng.uniform(-.27, .27)
        self.branch_onset = self.rise + rng.uniform(.19, .44) * self.stop
        self.span = (S[self.branch_start:] - S[self.branch_start]) / (1 - S[self.branch_start])
        self.branch_taper = np.maximum(0, np.sin(np.pi * self.span)) ** .78
        self.branch_tint = smoothstep(.64, 1, self.span)
        self.old_tail = smoothstep(S[self.branch_start], S[self.branch_start] + .14, S)
        self.bends = TravelingBends(rng, self.stop, 8)
        self.branch_bends = TravelingBends(rng, self.stop - self.branch_onset, 5)

    def channel(self, age):
        swing, depth = self.bends.sample(S, age)
        wanders = self.spin * age + .034 * np.sin(self.phase + age * 9)
        anchor = wanders + .17 * swing[-1] / (RADIAL[-1] + 60)
        angle = self.angle + anchor * S ** 1.7
        angle += TAPER * (self.curve + swing) / (RADIAL + 48)
        lift = max(0, age - .14) * .13 * np.sin(self.angle)
        elevation = self.elevation + TAPER * (self.depth + depth) + lift * S
        return angle, elevation

    def draw_branch(self, field, angle, elevation, age, parent_energy, fork_age):
        if self.split < .23 or fork_age < 0 or age > self.stop + .21:
            return

        start = self.branch_start
        span = self.span
        bend, depth = self.branch_bends.sample(span, fork_age)
        radius = RADIAL[start:]
        anchor = angle[-1] + self.branch_angle * (1 + .33 * fork_age)
        end_depth = np.clip(elevation[-1] + self.branch_depth, -1.06, 1.06)
        direction = angle[start] + (anchor - angle[start]) * span
        direction += self.branch_taper * bend / (radius + 48)
        inclination = elevation[start] + (end_depth - elevation[start]) * span
        inclination += depth * self.branch_taper
        branch = project(radius, direction, inclination)

        stop = self.stop - self.branch_onset + .10
        branch_gain = self.strength * (.97 if self.switch_route else .66)
        power = filament_light(span, fork_age, .083, stop, self.phase + 1.3, branch_gain)
        tint = COLD + self.branch_tint[:, None] * (ROSE - COLD)
        field.path(branch, power * parent_energy, tint, self.grain[start:][::-1])
        landed = smoothstep(.08, .14, fork_age) * (1 - smoothstep(stop, stop + .16, fork_age))
        if landed > .025:
            light_on_glass(field, anchor, end_depth, np.sign(self.spin), landed * branch_gain * .77, age)

    def draw(self, field, time):
        age = time - self.born
        if age < 0 or age > self.stop + .26:
            return
        angle, elevation = self.channel(age)
        path = project(RADIAL, angle, elevation)
        gain = self.strength * (1.08 if elevation[-1] > 0 else .91)
        energy = filament_light(S, age, self.rise, self.stop, self.phase, gain)
        fork_age = age - self.branch_onset

        takeover = smoothstep(.068, .17, fork_age) if self.switch_route else 0
        # The shared stem stays lit while the old route pinches off past the fork.
        energy *= 1 - .96 * takeover * self.old_tail
        core = COLD + smoothstep(.57, 1.04, energy)[:, None] * (WHITE - COLD)
        arrived = smoothstep(self.rise * .74, self.rise + .036, age)
        tint = core + (self.mixes * arrived)[:, None] * (ROSE - core)
        field.path(path, energy, tint, self.grain)

        afterglow = 1 - smoothstep(self.stop, self.stop + .24, age)
        flash = .62 + .38 * np.exp(-((age - self.rise) / .078) ** 2)
        remnant = 1 - smoothstep(.12, .30, fork_age) if self.switch_route else 1
        glow = arrived * afterglow * flash * gain * remnant
        if glow > .025:
            light_on_glass(field, angle[-1], elevation[-1], np.sign(self.spin), glow, age)
        root_power = .8 + .2 * min(1, energy[self.branch_start])
        self.draw_branch(field, angle, elevation, age, root_power, fork_age)


def generate_discharges(rng):
    channels = []
    for base in np.linspace(-math.pi, math.pi, 14, endpoint=False):
        born = rng.uniform(-1.1, 0)
        while born < DURATION:
            channel = Discharge(rng, born, base)
            channels.append(channel)
            born += channel.stop + rng.uniform(.045, .35)
    return channels


class GlyphField:
    def __init__(self, marks, strength, color):
        self.marks = marks.copy()
        self.strength = strength.copy()
        self.color = color.copy()
        self.spine = np.zeros((SPINE_HEIGHT, SPINE_WIDTH, 3), dtype=np.float32)

    def region(self, values, tint, glyphs):
        show = values > self.strength
        self.strength[show] = values[show]
        self.color[show] = (values[..., None] * tint)[show]
        self.marks[show] = glyphs[show] if isinstance(glyphs, np.ndarray) else glyphs

    def paint_spine(self, points, energy, tint):
        location = points / SPINE_SCALE - .5
        pixel = location.astype(np.intp)
        offset = location - pixel
        color = tint * energy[:, None]
        for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
            weight = (1 - np.abs(dx - offset[:, 0])) * (1 - np.abs(dy - offset[:, 1]))
            np.maximum.at(self.spine, (pixel[:, 1] + dy, pixel[:, 0] + dx),
                          color * weight[:, None])

    def path(self, points, energy, tint, grain):
        bend = np.gradient(points, axis=0)
        solid = tangent_glyphs(bend[:, 0], bend[:, 1])
        glyphs = np.select(
            (grain > .981, grain < .085, grain < .155),
            (MARK["*"], MARK["."], MARK[":"]),
            default=solid,
        )
        show = np.hypot(points[:, 0] - CENTER[0], points[:, 1] - CENTER[1]) < RADIUS + 4
        self.paint_spine(points[show], np.minimum(energy[show], 1), tint[show])
        x = np.int32(points[show, 0] / CELL_W)
        y = np.int32(points[show, 1] / CELL_H)
        # Resolve crossings by keeping the strongest filament in each character cell.
        index = y * COLS + x
        order = np.argsort(-energy[show], kind="stable")
        _, first = np.unique(index[order], return_index=True)
        best = order[first]
        x, y = x[best], y[best]
        power = np.clip(energy[show][best], 0, 1.08)
        visible = power > self.strength[y, x]
        y, x = y[visible], x[visible]
        self.strength[y, x] = power[visible]
        self.marks[y, x] = glyphs[show][best][visible]
        self.color[y, x] = power[visible, None] * tint[show][best][visible]


def paint_globe(texture, field):
    ring = np.exp(-((DISTANCE - RADIUS) / 4.5) ** 2)
    ripples = .72 + .20 * np.cos(ANGLE * 11 + .7) + .08 * np.sin(ANGLE * 27)
    side = .18 + .50 * ellipse_glow(ANGLE + 2.13, 0, .57, 1)
    side += .30 * ellipse_glow(ANGLE - .65, 0, .42, 1)
    side += .28 * ellipse_glow(ANGLE + .35, 0, .36, 1)
    silhouette = (ring * side * ripples * (.5 + .5 * texture)).astype(np.float32)
    field.region(silhouette, np.float32((.69, .62, 1)), tangent_glyphs(-DY, DX))

    reflection = np.exp(-((DISTANCE - RADIUS + 16 + 6 * np.cos(2.5 * ANGLE)) / 7) ** 2)
    reflection *= .39 * np.exp(-((ANGLE + 2.10) / .31) ** 2)
    field.region(reflection * (.35 + .65 * texture), np.float32((.75, .69, 1)), MARK[":"])
    glass_light = ellipse_glow(DX + 231, DY + 237, 132, 40)
    glass_light *= np.exp(-((DX - .86 * DY) / 49) ** 2) * (DISTANCE < RADIUS - 5)
    field.region(glass_light * texture * .13, np.float32((.59, .57, .93)), MARK["."])


def paint_fixture(texture, field):
    column_width = 27 + 4 * smoothstep(40, 400, DY)
    support = (np.abs(DX) < column_width) & (DY > 28) & (DY < 424)
    support_light = .036 + .096 * np.exp(-((DX + 15) / 5) ** 2)
    support_light *= .61 + .39 * np.cos(DY * .024) ** 2
    field.region(support * support_light * (.65 + .35 * texture), COLD, MARK["|"])

    tray = np.abs(DX) / 279
    curve = np.sqrt(np.maximum(0, 1 - tray ** 2))
    back, front = 1068 - 22 * curve, 1068 + 26 * curve
    bottom = 1107 + 28 * curve
    face = (CY > front) & (CY <= bottom) & (tray < 1)
    cap = (CY >= back) & (CY <= front) & (tray < 1)
    rim = np.exp(-((CY - front) / 5) ** 2) * (tray < 1)
    lip = np.exp(-((CY - back) / 6) ** 2) * (tray < 1)
    lower = np.exp(-((CY - bottom) / 6.4) ** 2) * (tray < 1)
    sheen = (.18 * rim + .15 * lip + .11 * lower + .079 * face + .045 * cap) * (.56 + .44 * texture)
    sheen += face * .14 * np.exp(-((DX + 95) / 83) ** 2) * texture
    field.region(sheen, np.float32((.64, .56, .86)), np.where(texture < .62, MARK["-"], MARK["."]))


class PlasmaBall:
    def __init__(self):
        rng = np.random.default_rng(71142)
        self.texture = rng.random((ROWS, COLS), dtype=np.float32)
        self.channels = generate_discharges(rng)
        self.root_glow = ellipse_glow(DX, DY, 77, 74)
        self.electrode = np.maximum(0, 1 - (DISTANCE / 57) ** 2)
        self.inner_rim = np.exp(-((DISTANCE - 51) / 6) ** 2)
        self.electrode_marks = np.select(
            (self.texture > .91, self.texture > .62, self.texture > .32),
            (MARK["+"], MARK[":"], MARK["-"]),
            default=MARK["."],
        )
        blank = GlyphField(
            np.zeros((ROWS, COLS), np.uint8),
            np.zeros((ROWS, COLS), np.float32),
            np.zeros((ROWS, COLS, 3), np.float32),
        )
        paint_globe(self.texture, blank)
        paint_fixture(self.texture, blank)
        self.base = blank
        self.font = mono(23)
        self.panel = CodePanel()
        self.atlas = make_glyph_atlas()
        glass = ellipse_glow(DX + 78, DY + 126, 412, 358)
        ring = np.exp(-((DISTANCE - RADIUS) / 15) ** 2)
        ring *= .22 + .78 * np.exp(-((ANGLE + 2.05) / .73) ** 2)
        self.background = np.zeros((ROWS, COLS, 3), dtype=np.float32)
        self.background[:] = (2.0, 1.5, 5.0)
        self.background += glass[..., None] * np.float32((4, 1.4, 7))
        self.background += ring[..., None] * np.float32((15, 7, 19))
        cast = ellipse_glow(DX, CY - 1142, 298, 35)
        self.background += cast[..., None] * np.float32((4, 2, 8))

    def paint_electrode(self, field, time):
        wave = np.sin(.060 * DY + 2.3 * ANGLE + time * 2.9)
        wave += .58 * np.sin(DX * .105 - .085 * DY + 3.8 * time)
        bands = smoothstep(-1.29, 1.5, wave)
        body = self.electrode * (.22 + .54 * bands)
        edge = self.inner_rim * (.34 + .54 * np.sin(2.3 * time + 5 * ANGLE) ** 2)
        light = (body + edge) * (.68 + .32 * self.texture)
        field.region(light, ROSE, self.electrode_marks)

    def render(self, time):
        field = GlyphField(self.base.marks, self.base.strength, self.base.color)
        self.paint_electrode(field, time)
        for channel in self.channels:
            channel.draw(field, time)

        haze = cv2.GaussianBlur(field.color, (0, 0), 2.1)
        halo = cv2.GaussianBlur(field.color, (0, 0), .73)
        room = self.background + 128 * haze + 101 * halo
        room += self.root_glow[..., None] * np.float32((29, 6, 35))
        backing = cv2.resize(np.uint8(np.clip(room, 0, 255)), (WIDTH, ART_HEIGHT), interpolation=cv2.INTER_LINEAR)

        ink = self.atlas[field.marks][..., None] * field.color[:, :, None, None, :]
        ink = np.uint8(np.minimum(ink * 255, 255)).transpose(0, 2, 1, 3, 4).reshape(ART_HEIGHT, WIDTH, 3)
        spine = cv2.resize(np.uint8(np.clip(66 * field.spine, 0, 255)),
                           (WIDTH, ART_HEIGHT), interpolation=cv2.INTER_LINEAR)
        pixels = cv2.add(cv2.add(backing, spine), ink)
        canvas = Image.new("RGB", (WIDTH, HEIGHT), "#0b0b13")
        canvas.paste(Image.fromarray(pixels), (0, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((33, 29), "plasma_ball.py", font=self.font, fill="#c7bfdc")
        draw.text((WIDTH - 33, 29), f"t {time:05.2f}s", font=self.font, fill="#b4abc8", anchor="ra")
        self.panel.draw(canvas, time)
        return canvas


def mono(size):
    for name in ("consola.ttf", "DejaVuSansMono.ttf", "LiberationMono-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size)


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
        for function in (TravelingBends.sample, Discharge.channel, Discharge.draw_branch, Discharge.draw):
            lines, line_number = inspect.getsourcelines(function)
            start = max(0, line_number - 4)
            end = start + len(lines) + 15
            rows = []
            for number, line in enumerate(source[start:end], start + 1):
                indent = " " * (len(line) - len(line.lstrip()))
                wrapped = textwrap.wrap(
                    line.lstrip(), width=64, initial_indent=indent, subsequent_indent=indent + "    ",
                    break_long_words=False, break_on_hyphens=False,
                ) or [""]
                rows.append((str(number), wrapped[0]))
                rows.extend(("", continuation) for continuation in wrapped[1:])
            body = Image.new("RGB", (PANEL_W, len(rows) * LINE_HEIGHT), "#110e19")
            draw = ImageDraw.Draw(body)
            for index, (number, line) in enumerate(rows):
                y = index * LINE_HEIGHT + 5
                draw.text((51, y), number, font=self.font, fill="#758094", anchor="rt")
                self.draw_line(draw, line, y)
            scrollable = sum(not number or int(number) < line_number + len(lines) for number, _ in rows)
            self.sections.append((body, scrollable))
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
        panel = Image.new("RGB", (PANEL_W, PANEL_H), "#110e19")
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
        draw.line((0, HEADER_H - 1, PANEL_W, HEADER_H - 1), fill="#30233f", width=2)
        for index, color in enumerate(("#ff665c", "#ffbe4c", "#38c66d")):
            left = 18 + index * 24
            draw.ellipse((left, 17, left + 13, 30), fill=color)
        draw.text((91, 23), "plasma_ball.py", font=self.font, fill="#d6dbe3", anchor="lm")
        draw.text((1001, 23), f"running   {FPS} fps", font=self.font, fill="#a4adbe", anchor="rm")
        draw.ellipse((763, 18, 774, 29), fill="#52ce73")
        draw.rounded_rectangle((0, 0, PANEL_W - 1, PANEL_H - 1), radius=17, outline="#382947", width=2)
        frame.paste(panel, (30, 1364), self.mask)


def make_glyph_atlas():
    font = mono(16)
    masks = []
    for character in GLYPHS:
        mask = Image.new("L", (CELL_W, CELL_H))
        draw = ImageDraw.Draw(mask)
        left, top, right, bottom = font.getbbox(character)
        origin = ((CELL_W - right + left) / 2 - left, (CELL_H - bottom + top) / 2 - top)
        draw.text(origin, character, font=font, fill=255)
        if character == "-":
            draw.line((2, CELL_H // 2, CELL_W - 3, CELL_H // 2), fill=241)
        masks.append(np.asarray(mask, dtype=np.float32) / 255)
    return np.stack(masks)


def make_preview(film, output):
    times = (0.57, 1.47, 2.55, 5.28, 8.50, 11.35, 15.60, 19.55)
    sheet = Image.new("RGB", (1080, 1010), "#14111d")
    draw = ImageDraw.Draw(sheet)
    for index, time in enumerate(times):
        frame = film.render(time)
        x, y = index % 4 * 270, index // 4 * 503
        draw.text((x + 16, y + 4), f"{time:05.2f}s", fill="white")
        sheet.paste(frame.resize((270, 480), Image.Resampling.LANCZOS), (x, y + 20))
        if index in (0, 3, 5):
            frame.crop((103, 224, 1037, 1150)).save(output.with_name(f"reroute-detail-{index}.png"))
    sheet.save(output, quality=94)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    cv2.setNumThreads(2)
    film = PlasmaBall()
    if args.preview:
        make_preview(film, args.output)
        return

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg is required to encode the video.")
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-n", "-f", "rawvideo",
        "-pixel_format", "rgb24", "-video_size", f"{WIDTH}x{HEIGHT}", "-framerate", str(FPS),
        "-i", "-", "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "15",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-metadata", "comment=Procedural ASCII plasma globe", str(args.output),
    ]
    assert film.atlas.shape == (len(GLYPHS), CELL_H, CELL_W)
    with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
        try:
            for index in range(DURATION * FPS):
                encoder.stdin.write(film.render(index / FPS).tobytes())
                if (index + 1) % 150 == 0:
                    print(f"{(index + 1) // FPS}/{DURATION} seconds rendered", flush=True)
        finally:
            encoder.stdin.close()
        if encoder.wait():
            raise RuntimeError("FFmpeg could not encode the plasma globe.")


if __name__ == "__main__":
    main()

