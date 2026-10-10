#include "pch.h"
#include "OrtEnv.hpp"

#include <dml_provider_factory.h>
#include <onnxruntime_cxx_api.h>

namespace NeuralModelIntegrateTestbed {

void SetOrtError(std::string* error, std::string message) {
    if (error != nullptr) {
        *error = std::move(message);
    }
}

Ort::Env& SharedOrtEnv() {
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ptflio");
    return env;
}

const OrtDmlApi* SharedDmlApi(std::string* error) {
    static const OrtDmlApi* api = [] {
        const OrtDmlApi* result = nullptr;
        const OrtStatus* status = Ort::GetApi().GetExecutionProviderApi(
            "DML", ORT_API_VERSION, reinterpret_cast<const void**>(&result));
        if (status != nullptr) {
            Ort::GetApi().ReleaseStatus(const_cast<OrtStatus*>(status));
            return static_cast<const OrtDmlApi*>(nullptr);
        }
        return result;
    }();
    if (api == nullptr) {
        SetOrtError(error, "ONNX Runtime has no DirectML execution provider; the "
                           "onnxruntime.dll beside the executable is probably not "
                           "the DirectML build");
    }
    return api;
}

}  // namespace NeuralModelIntegrateTestbed
