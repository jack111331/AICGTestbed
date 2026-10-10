#pragma once

#include <string>

// ONNX Runtime pieces that must be shared across every session in the process.
//
// This header deliberately does NOT include onnxruntime_cxx_api.h: it is
// included by translation units that only need to report an error, and ORT's
// headers are large. The definitions live in OrtEnv.cc, which is the only place
// that needs the real types.
namespace Ort {
struct Env;
}
struct OrtDmlApi;

namespace NeuralModelIntegrateTestbed {

// One Ort::Env per process. ORT keys its logger and thread pools off this, so
// creating one per model would spin up a set each time -- and the FloodDiffusion
// pipeline alone holds four sessions.
Ort::Env& SharedOrtEnv();

// The DirectML-specific function table, fetched once. Null (with `error` filled)
// when the onnxruntime.dll beside the executable is not the DirectML build.
const OrtDmlApi* SharedDmlApi(std::string* error);

// Assigns `message` when `error` is non-null. Every entry point in the ONNX
// Runtime wrappers reports failure this way rather than throwing, because they
// are driven from a frame loop.
void SetOrtError(std::string* error, std::string message);

}  // namespace NeuralModelIntegrateTestbed
