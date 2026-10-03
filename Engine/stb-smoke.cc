// Smoke test for //third:stb.
//
// stb_image is header-only with a separate implementation macro, so the thing
// actually worth proving is that the single STB_IMAGE_IMPLEMENTATION translation
// unit (third/stb_impl.cc) linked in and decodes correctly. Decodes a known
// 4x2 RGBA PNG two ways -- from memory and from a file, exercising the stdio
// path -- and checks the pixels rather than just a non-null return.
#include <stb_image.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

namespace {

// 4x2 8-bit RGBA PNG. Row 0: red, green, blue, white. Row 1: black, yellow,
// cyan, magenta.
const unsigned char kPng[] = {
    0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d,
    0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x02,
    0x08, 0x06, 0x00, 0x00, 0x00, 0x7f, 0xa8, 0x7d, 0x63, 0x00, 0x00, 0x00,
    0x1a, 0x49, 0x44, 0x41, 0x54, 0x78, 0xda, 0x63, 0xf8, 0xcf, 0xc0, 0xf0,
    0x1f, 0x0c, 0x19, 0xfe, 0x83, 0x01, 0x03, 0x03, 0x98, 0x05, 0x12, 0x02,
    0x53, 0xff, 0x01, 0x34, 0xfb, 0x13, 0xed, 0x5a, 0xe5, 0x90, 0xc3, 0x00,
    0x00, 0x00, 0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
};

struct Pixel {
  int x, y;
  unsigned char r, g, b, a;
};

const Pixel kExpected[] = {
    {0, 0, 255, 0, 0, 255},      {1, 0, 0, 255, 0, 255},
    {3, 0, 255, 255, 255, 255},  {0, 1, 0, 0, 0, 255},
    {2, 1, 0, 255, 255, 255},    {3, 1, 255, 0, 255, 255},
};

bool CheckPixels(const char* label, const unsigned char* pixels, int w) {
  bool ok = true;
  for (const Pixel& p : kExpected) {
    const unsigned char* got = pixels + (p.y * w + p.x) * 4;
    if (got[0] != p.r || got[1] != p.g || got[2] != p.b || got[3] != p.a) {
      std::fprintf(stderr,
                   "%s: pixel (%d,%d) = %u,%u,%u,%u expected %u,%u,%u,%u\n",
                   label, p.x, p.y, got[0], got[1], got[2], got[3], p.r, p.g,
                   p.b, p.a);
      ok = false;
    }
  }
  return ok;
}

}  // namespace

int main() {
  std::cout << "stb_image linked OK" << std::endl;

  // stbi_info first: reports dimensions without decoding.
  int iw = 0, ih = 0, ic = 0;
  if (!stbi_info_from_memory(kPng, static_cast<int>(sizeof(kPng)), &iw, &ih,
                             &ic)) {
    std::cerr << "stbi_info_from_memory failed: " << stbi_failure_reason()
              << std::endl;
    return 1;
  }
  std::cout << "  info:        " << iw << "x" << ih << " ch=" << ic
            << std::endl;

  // Path 1: decode straight from the embedded bytes.
  int w = 0, h = 0, channels = 0;
  unsigned char* from_memory = stbi_load_from_memory(
      kPng, static_cast<int>(sizeof(kPng)), &w, &h, &channels, 4);
  if (from_memory == nullptr) {
    std::cerr << "stbi_load_from_memory failed: " << stbi_failure_reason()
              << std::endl;
    return 1;
  }
  const bool memory_ok = w == 4 && h == 2 && CheckPixels("memory", from_memory, w);
  std::cout << "  from memory: " << w << "x" << h << " ch=" << channels
            << " pixels=" << (memory_ok ? "match" : "MISMATCH") << std::endl;
  stbi_image_free(from_memory);

  // Path 2: same bytes via a real file, which exercises stb's stdio code.
  std::string path;
  if (const char* tmp = std::getenv("TEST_TMPDIR")) {
    path = std::string(tmp) + "/stb-smoke.png";
  } else if (const char* temp = std::getenv("TEMP")) {
    path = std::string(temp) + "/stb-smoke.png";
  } else {
    path = "stb-smoke.png";
  }
  {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(kPng), sizeof(kPng));
    if (!out) {
      std::cerr << "could not write " << path << std::endl;
      return 1;
    }
  }

  int fw = 0, fh = 0, fchannels = 0;
  unsigned char* from_file = stbi_load(path.c_str(), &fw, &fh, &fchannels, 4);
  if (from_file == nullptr) {
    std::cerr << "stbi_load(" << path << ") failed: " << stbi_failure_reason()
              << std::endl;
    return 1;
  }
  const bool file_ok = fw == 4 && fh == 2 && CheckPixels("file", from_file, fw);
  std::cout << "  from file:   " << fw << "x" << fh << " ch=" << fchannels
            << " pixels=" << (file_ok ? "match" : "MISMATCH") << std::endl;
  stbi_image_free(from_file);
  std::remove(path.c_str());

  return (memory_ok && file_ok) ? 0 : 1;
}
