# Writes a Radiance .hdr equirectangular test environment whose colours encode
# direction, so a readback can check the skybox's mapping instead of trusting it.
#
# Longitude is split into four quadrants and the poles are marked:
#
#   u 0.375..0.625  (centre, looking down -Z)  GREEN
#   u 0.625..0.875  (looking down +X)          RED
#   u 0.875..0.125  (looking down +Z)          YELLOW
#   u 0.125..0.375  (looking down -X)          BLUE
#   v < 0.08        (up, +Y)                   WHITE
#   v > 0.92        (down, -Y)                 GREY
#
# Values are low and exact in float so an exposure of 1.0 reproduces them
# byte-for-byte in an 8-bit back buffer.
import math
import struct
import sys

WIDTH, HEIGHT = 256, 128
# Default output sits where the app's skybox search will find it:
#
#     python tools/make_env_fixture.py resources/alpha-test/env-test.hdr
#     PTFLIO_SCENE=resources/alpha-test/alpha-test.gltf \
#         PTFLIO_SKYBOX=resources/alpha-test/env-test.hdr bazel-bin/Engine/hello-world.exe
#
# Point the camera down -Z and the middle of the screen must be GREEN, the
# right RED, the left BLUE, behind you YELLOW. Anything else means the
# equirectangular mapping in PSSkybox is wrong.
#
# Check the HUE, not the value: LoadEnvironment sets exposure from the image's
# median luminance, so these colours arrive scaled (0.503 for this fixture).
# Exposure is a scalar, so it cannot change which quadrant you are looking at.
OUT = sys.argv[1] if len(sys.argv) > 1 else "env-test.hdr"

GREEN = (0.0, 0.5, 0.0)
RED = (0.5, 0.0, 0.0)
YELLOW = (0.5, 0.5, 0.0)
BLUE = (0.0, 0.0, 0.5)
WHITE = (0.75, 0.75, 0.75)
GREY = (0.25, 0.25, 0.25)


def colour(u, v):
    if v < 0.08:
        return WHITE
    if v > 0.92:
        return GREY
    # Quadrant centres at u = 0.5 (-Z), 0.75 (+X), 0.0/1.0 (+Z), 0.25 (-X).
    if 0.375 <= u < 0.625:
        return GREEN
    if 0.625 <= u < 0.875:
        return RED
    if 0.125 <= u < 0.375:
        return BLUE
    return YELLOW


def rgbe(r, g, b):
    peak = max(r, g, b)
    if peak < 1e-32:
        return bytes((0, 0, 0, 0))
    mantissa, exponent = math.frexp(peak)
    scale = mantissa * 256.0 / peak
    return bytes((
        min(255, int(r * scale)),
        min(255, int(g * scale)),
        min(255, int(b * scale)),
        exponent + 128,
    ))


rows = []
for y in range(HEIGHT):
    v = (y + 0.5) / HEIGHT
    row = bytearray()
    for x in range(WIDTH):
        u = (x + 0.5) / WIDTH
        row += rgbe(*colour(u, v))
    rows.append(bytes(row))

with open(OUT, "wb") as handle:
    handle.write(b"#?RADIANCE\n")
    handle.write(b"FORMAT=32-bit_rle_rgbe\n")
    handle.write(b"\n")
    # -Y rows +X columns: the first scanline is the TOP of the image, which is
    # v = 0, which this mapping puts at +Y. Matches acos(dir.y)/pi.
    handle.write(("-Y %d +X %d\n" % (HEIGHT, WIDTH)).encode("ascii"))
    for row in rows:
        handle.write(row)

print("wrote", OUT, "%dx%d" % (WIDTH, HEIGHT))
print("expect, at yaw 0 with the camera looking down -Z: GREEN ahead,")
print("RED to the right (+X), BLUE to the left (-X), YELLOW behind (+Z)")
