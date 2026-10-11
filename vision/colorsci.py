"""Perceptual color distance (LAB + CIEDE2000) -- pure Python, no deps.

After many rounds of hand-tuned HSV/RGB thresholds failing on edge cases
(orange vs red, green vs white, logo colors), we switch to the industry-standard
perceptual approach used by the Qbr project:

  * convert colors to CIE L*a*b*, then
  * measure difference with CIEDE2000 (dE00), which models how different two
    colors look to a human eye.

Combined with calibrating on the user's OWN six center colors (see
ColorClassifier.from_palette), classification compares real measured colors to
real measured colors with a perceptual metric -- far more robust than absolute
thresholds under odd lighting.

Attribution: bgr2lab adapted from a widely-used sRGB->LAB routine
(stackoverflow.com/a/16020102); ciede2000 adapted from
https://github.com/lovro-i/CIEDE2000. Both as used by the Qbr project
(https://github.com/kkoomen/qbr, MIT, (c) 2016 Kim Koomen).
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

RGB = Tuple[float, float, float]
Lab = Tuple[float, float, float]


def rgb_to_lab(rgb: RGB) -> Lab:
    """Convert an sRGB color (0..255) to CIE L*a*b*."""
    out = []
    for value in rgb:
        value = float(value) / 255.0
        if value > 0.04045:
            value = ((value + 0.055) / 1.055) ** 2.4
        else:
            value = value / 12.92
        out.append(value * 100.0)
    r, g, b = out

    X = r * 0.4124 + g * 0.3576 + b * 0.1805
    Y = r * 0.2126 + g * 0.7152 + b * 0.0722
    Z = r * 0.0193 + g * 0.1192 + b * 0.9505

    # Reference white D65, 2 degree observer.
    xyz = [X / 95.047, Y / 100.0, Z / 108.883]
    for i, value in enumerate(xyz):
        if value > 0.008856:
            value = value ** (1.0 / 3.0)
        else:
            value = (7.787 * value) + (16.0 / 116.0)
        xyz[i] = value
    x, y, z = xyz

    L = (116.0 * y) - 16.0
    a = 500.0 * (x - y)
    bb = 200.0 * (y - z)
    return (L, a, bb)


def ciede2000(lab1: Lab, lab2: Lab) -> float:
    """CIEDE2000 (dE00) perceptual color difference between two LAB colors.

    Adapted from https://github.com/lovro-i/CIEDE2000.
    """
    C_25_7 = 6103515625.0  # 25**7

    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    C1 = math.sqrt(a1 ** 2 + b1 ** 2)
    C2 = math.sqrt(a2 ** 2 + b2 ** 2)
    C_ave = (C1 + C2) / 2.0
    G = 0.5 * (1 - math.sqrt(C_ave ** 7 / (C_ave ** 7 + C_25_7)))

    L1_, L2_ = L1, L2
    a1_, a2_ = (1 + G) * a1, (1 + G) * a2
    b1_, b2_ = b1, b2

    C1_ = math.sqrt(a1_ ** 2 + b1_ ** 2)
    C2_ = math.sqrt(a2_ ** 2 + b2_ ** 2)

    if b1_ == 0 and a1_ == 0:
        h1_ = 0
    elif a1_ >= 0:
        h1_ = math.atan2(b1_, a1_)
    else:
        h1_ = math.atan2(b1_, a1_) + 2 * math.pi

    if b2_ == 0 and a2_ == 0:
        h2_ = 0
    elif a2_ >= 0:
        h2_ = math.atan2(b2_, a2_)
    else:
        h2_ = math.atan2(b2_, a2_) + 2 * math.pi

    dL_ = L2_ - L1_
    dC_ = C2_ - C1_
    dh_ = h2_ - h1_
    if C1_ * C2_ == 0:
        dh_ = 0
    elif dh_ > math.pi:
        dh_ -= 2 * math.pi
    elif dh_ < -math.pi:
        dh_ += 2 * math.pi
    dH_ = 2 * math.sqrt(C1_ * C2_) * math.sin(dh_ / 2.0)

    L_ave = (L1_ + L2_) / 2.0
    C_ave = (C1_ + C2_) / 2.0

    _dh = abs(h1_ - h2_)
    _sh = h1_ + h2_
    C1C2 = C1_ * C2_

    if _dh <= math.pi and C1C2 != 0:
        h_ave = (h1_ + h2_) / 2.0
    elif _dh > math.pi and _sh < 2 * math.pi and C1C2 != 0:
        h_ave = (h1_ + h2_) / 2.0 + math.pi
    elif _dh > math.pi and _sh >= 2 * math.pi and C1C2 != 0:
        h_ave = (h1_ + h2_) / 2.0 - math.pi
    else:
        h_ave = h1_ + h2_

    T = (1 - 0.17 * math.cos(h_ave - math.pi / 6)
         + 0.24 * math.cos(2 * h_ave)
         + 0.32 * math.cos(3 * h_ave + math.pi / 30)
         - 0.20 * math.cos(4 * h_ave - 63 * math.pi / 180))

    h_ave_deg = h_ave * 180 / math.pi
    if h_ave_deg < 0:
        h_ave_deg += 360
    elif h_ave_deg > 360:
        h_ave_deg -= 360
    dTheta = 30 * math.exp(-(((h_ave_deg - 275) / 25) ** 2))

    R_C = 2 * math.sqrt(C_ave ** 7 / (C_ave ** 7 + C_25_7))
    S_C = 1 + 0.045 * C_ave
    S_H = 1 + 0.015 * C_ave * T

    Lm50s = (L_ave - 50) ** 2
    S_L = 1 + 0.015 * Lm50s / math.sqrt(20 + Lm50s)
    R_T = -math.sin(dTheta * math.pi / 90) * R_C

    f_L = dL_ / S_L
    f_C = dC_ / S_C
    f_H = dH_ / S_H

    return math.sqrt(f_L ** 2 + f_C ** 2 + f_H ** 2 + R_T * f_C * f_H)


def rgb_distance(a: RGB, b: RGB) -> float:
    """Perceptual distance between two sRGB colors via LAB + CIEDE2000."""
    return ciede2000(rgb_to_lab(a), rgb_to_lab(b))
