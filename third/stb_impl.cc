// The one and only translation unit that compiles stb_image's implementation.
//
// stb_image.h is both header and source: it emits function bodies only when
// STB_IMAGE_IMPLEMENTATION is defined. Defining it in two TUs gives duplicate
// symbols at link time, which is why //third:stb sets it via local_defines (so
// it cannot leak to dependents) and why this file exists instead of the macro
// being sprinkled through Engine/.
//
// Callers just depend on //third:stb and #include <stb_image.h>.
//
// To add another stb library (stb_image_write for screenshots,
// stb_image_resize2 for mip chains), add its *_IMPLEMENTATION macro to
// local_defines in //third:stb and include its header below.

#include <stb_image.h>
