#include "pch.h"
#include "OnnxModel.hpp"

#include <onnx/onnx_pb.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace NeuralModelIntegrateTestbed {

struct OnnxModel::Asset {
    onnx::ModelProto model;
};

namespace {

[[noreturn]] void Fail(const std::string& message) {
    throw std::runtime_error(message);
}

// ONNX element type -> DirectML element type. Only the types DirectML actually
// has are mapped; anything else is named in the error rather than silently
// reinterpreted, which would read the weight bytes at the wrong stride.
bool ToDmlDataType(int onnxType, DML_TENSOR_DATA_TYPE& out) {
    switch (onnxType) {
        case onnx::TensorProto_DataType_FLOAT:   out = DML_TENSOR_DATA_TYPE_FLOAT32; return true;
        case onnx::TensorProto_DataType_FLOAT16: out = DML_TENSOR_DATA_TYPE_FLOAT16; return true;
        case onnx::TensorProto_DataType_DOUBLE:  out = DML_TENSOR_DATA_TYPE_FLOAT64; return true;
        case onnx::TensorProto_DataType_UINT8:   out = DML_TENSOR_DATA_TYPE_UINT8;   return true;
        case onnx::TensorProto_DataType_INT8:    out = DML_TENSOR_DATA_TYPE_INT8;    return true;
        case onnx::TensorProto_DataType_UINT16:  out = DML_TENSOR_DATA_TYPE_UINT16;  return true;
        case onnx::TensorProto_DataType_INT16:   out = DML_TENSOR_DATA_TYPE_INT16;   return true;
        case onnx::TensorProto_DataType_INT32:   out = DML_TENSOR_DATA_TYPE_INT32;   return true;
        case onnx::TensorProto_DataType_UINT32:  out = DML_TENSOR_DATA_TYPE_UINT32;  return true;
        case onnx::TensorProto_DataType_INT64:   out = DML_TENSOR_DATA_TYPE_INT64;   return true;
        case onnx::TensorProto_DataType_UINT64:  out = DML_TENSOR_DATA_TYPE_UINT64;  return true;
        default: return false;
    }
}

std::string OnnxDataTypeName(int onnxType) {
    const auto* descriptor = onnx::TensorProto_DataType_descriptor();
    const auto* value = descriptor->FindValueByNumber(onnxType);
    if (value == nullptr) {
        return "UNKNOWN(" + std::to_string(onnxType) + ")";
    }
    // protobuf 6.x returns a string_view here, which must be copied rather
    // than pointed at.
    const auto view = value->name();
    return std::string(view.data(), view.size());
}

// Attribute lookup. ONNX attributes are an unordered repeated field, so these
// are linear scans -- fine for the handful each operator has.
const onnx::AttributeProto* FindAttribute(const onnx::NodeProto& node, const char* name) {
    for (const onnx::AttributeProto& attr : node.attribute()) {
        if (attr.name() == name) {
            return &attr;
        }
    }
    return nullptr;
}

std::vector<uint32_t> AttributeInts(const onnx::NodeProto& node,
                                    const char* name,
                                    std::vector<uint32_t> fallback) {
    const onnx::AttributeProto* attr = FindAttribute(node, name);
    if (attr == nullptr || attr->ints_size() == 0) {
        return fallback;
    }
    std::vector<uint32_t> out;
    out.reserve(attr->ints_size());
    for (const int64_t value : attr->ints()) {
        if (value < 0) {
            Fail("node '" + node.name() + "' (" + node.op_type() + ") has a negative " +
                 name + ", which DirectML cannot express");
        }
        out.push_back(static_cast<uint32_t>(value));
    }
    return out;
}

float AttributeFloat(const onnx::NodeProto& node, const char* name, float fallback) {
    const onnx::AttributeProto* attr = FindAttribute(node, name);
    return attr == nullptr ? fallback : attr->f();
}

int64_t AttributeInt(const onnx::NodeProto& node, const char* name, int64_t fallback) {
    const onnx::AttributeProto* attr = FindAttribute(node, name);
    return attr == nullptr ? fallback : attr->i();
}

// An initializer's bytes. ONNX stores a tensor either as raw_data or in a
// type-specific repeated field, and real exporters use both -- raw_data for
// weights, the typed fields for small constants written by hand.
std::vector<uint8_t> InitializerBytes(const onnx::TensorProto& tensor,
                                      uint64_t expectedBytes) {
    std::vector<uint8_t> out;

    if (tensor.has_raw_data()) {
        const std::string& raw = tensor.raw_data();
        out.assign(raw.begin(), raw.end());
    } else if (tensor.float_data_size() > 0) {
        out.resize(static_cast<std::size_t>(tensor.float_data_size()) * sizeof(float));
        for (int i = 0; i < tensor.float_data_size(); ++i) {
            const float value = tensor.float_data(i);
            std::memcpy(out.data() + static_cast<std::size_t>(i) * sizeof(float), &value,
                        sizeof(float));
        }
    } else if (tensor.int64_data_size() > 0) {
        out.resize(static_cast<std::size_t>(tensor.int64_data_size()) * sizeof(int64_t));
        for (int i = 0; i < tensor.int64_data_size(); ++i) {
            const int64_t value = tensor.int64_data(i);
            std::memcpy(out.data() + static_cast<std::size_t>(i) * sizeof(int64_t), &value,
                        sizeof(int64_t));
        }
    } else if (tensor.int32_data_size() > 0) {
        out.resize(static_cast<std::size_t>(tensor.int32_data_size()) * sizeof(int32_t));
        for (int i = 0; i < tensor.int32_data_size(); ++i) {
            const int32_t value = tensor.int32_data(i);
            std::memcpy(out.data() + static_cast<std::size_t>(i) * sizeof(int32_t), &value,
                        sizeof(int32_t));
        }
    } else if (tensor.has_data_location() &&
               tensor.data_location() == onnx::TensorProto_DataLocation_EXTERNAL) {
        Fail("initializer '" + tensor.name() +
             "' stores its data externally, which is not loaded");
    }

    if (out.size() != expectedBytes) {
        Fail("initializer '" + tensor.name() + "' holds " + std::to_string(out.size()) +
             " bytes but its shape implies " + std::to_string(expectedBytes));
    }
    return out;
}

std::vector<uint32_t> TensorProtoDims(const onnx::TensorProto& tensor) {
    std::vector<uint32_t> sizes;
    sizes.reserve(tensor.dims_size());
    for (const int64_t dim : tensor.dims()) {
        if (dim <= 0) {
            Fail("initializer '" + tensor.name() + "' has a non-positive dimension");
        }
        sizes.push_back(static_cast<uint32_t>(dim));
    }
    // DirectML has no concept of a scalar tensor; a 0-D ONNX constant becomes
    // a single element.
    if (sizes.empty()) {
        sizes.push_back(1);
    }
    return sizes;
}

// Translating one node needs the running tensor table plus the ability to
// materialise an initializer on first use, which is what this carries.
struct TranslationContext {
    dml::Graph* graph = nullptr;
    const onnx::GraphProto* onnxGraph = nullptr;

    std::unordered_map<std::string, dml::Expression> tensors;
    std::map<std::string, const onnx::TensorProto*> initializers;

    // Weights get their graph input index as they are first used, so the
    // indices stay contiguous from 1 and unused initializers cost nothing.
    uint32_t nextInputIndex = 1;
    std::vector<ModelWeight>* weights = nullptr;

    dml::Expression Get(const std::string& name) {
        const auto it = tensors.find(name);
        if (it != tensors.end()) {
            return it->second;
        }
        const auto init = initializers.find(name);
        if (init == initializers.end()) {
            Fail("tensor '" + name + "' is used before it is produced, and is not an "
                 "initializer -- the graph may not be in topological order");
        }
        return Materialise(*init->second, {});
    }

    // Same, but forces the declared shape. Used for a Conv bias, which ONNX
    // stores as 1-D [M] while DirectML wants it rank-matched to the output as
    // [1, M, 1, 1]. The element count has to agree; only the shape changes.
    dml::Expression GetShaped(const std::string& name,
                              const dml::TensorDesc::Dimensions& sizes) {
        const auto it = tensors.find(name);
        if (it != tensors.end()) {
            // Already materialised, or computed by an earlier node: a view is
            // the only option left.
            return dml::Reinterpret(it->second, sizes, dml::NullOpt);
        }
        const auto init = initializers.find(name);
        if (init == initializers.end()) {
            Fail("tensor '" + name + "' is used before it is produced, and is not an "
                 "initializer");
        }
        return Materialise(*init->second, sizes);
    }

    bool Has(const std::string& name) const {
        return tensors.count(name) != 0 || initializers.count(name) != 0;
    }

    // Turns an ONNX initializer into a DirectML input tensor plus the bytes
    // ModelManager will upload. `shapeOverride`, when non-empty, replaces the
    // declared shape -- see GetShaped.
    dml::Expression Materialise(const onnx::TensorProto& tensor,
                                const dml::TensorDesc::Dimensions& shapeOverride) {
        DML_TENSOR_DATA_TYPE dataType;
        if (!ToDmlDataType(tensor.data_type(), dataType)) {
            Fail("initializer '" + tensor.name() + "' has element type " +
                 OnnxDataTypeName(tensor.data_type()) + ", which DirectML does not support");
        }

        dml::TensorDesc::Dimensions sizes = TensorProtoDims(tensor);
        if (!shapeOverride.empty()) {
            uint64_t declared = 1;
            for (const uint32_t dim : sizes) {
                declared *= dim;
            }
            uint64_t wanted = 1;
            for (const uint32_t dim : shapeOverride) {
                wanted *= dim;
            }
            if (declared != wanted) {
                Fail("initializer '" + tensor.name() + "' has " + std::to_string(declared) +
                     " elements but is needed as a shape holding " + std::to_string(wanted));
            }
            sizes = shapeOverride;
        }
        // OWNED_BY_DML lets DirectML keep its own reordered copy, which is why
        // these are bound to the operator initializer and not at execute time.
        const dml::TensorDesc desc(dataType, DML_TENSOR_FLAG_OWNED_BY_DML, sizes);

        const uint32_t index = nextInputIndex++;
        auto expression = dml::InputTensor(*graph, index, desc);

        weights->push_back(ModelWeight{tensor.name(), index, desc,
                                       InitializerBytes(tensor, desc.totalTensorSizeInBytes)});
        tensors.emplace(tensor.name(), expression);
        return expression;
    }
};

// --- operator handlers ------------------------------------------------------

dml::Expression TranslateConv(const onnx::NodeProto& node, TranslationContext& ctx) {
    if (node.input_size() < 2) {
        Fail("Conv node '" + node.name() + "' needs at least X and W");
    }
    auto input = ctx.Get(node.input(0));
    auto filter = ctx.Get(node.input(1));

    const auto filterSizes = filter.GetOutputDesc().sizes;
    const auto inputRank = input.GetOutputDesc().sizes.size();

    dml::Optional<dml::Expression> bias;
    if (node.input_size() >= 3 && !node.input(2).empty()) {
        if (filterSizes.empty()) {
            Fail("Conv node '" + node.name() + "' has an empty filter shape");
        }
        // [1, outputChannels, 1, ...]: DirectML wants the bias rank-matched to
        // the output, which ONNX's flat [M] is not.
        dml::TensorDesc::Dimensions biasSizes(inputRank, 1);
        if (inputRank >= 2) {
            biasSizes[1] = filterSizes[0];
        }
        bias = ctx.GetShaped(node.input(2), biasSizes);
    }
    if (filterSizes.size() < 3) {
        Fail("Conv node '" + node.name() + "' has a filter of rank " +
             std::to_string(filterSizes.size()) + "; 2-D convolution needs rank 4");
    }
    // Spatial rank, i.e. everything after [outChannels, inChannels].
    const std::size_t spatial = filterSizes.size() - 2;

    std::vector<uint32_t> kernelShape;
    for (std::size_t i = 0; i < spatial; ++i) {
        kernelShape.push_back(filterSizes[i + 2]);
    }

    const std::vector<uint32_t> strides =
        AttributeInts(node, "strides", std::vector<uint32_t>(spatial, 1));
    const std::vector<uint32_t> dilations =
        AttributeInts(node, "dilations", std::vector<uint32_t>(spatial, 1));

    // ONNX `pads` is [start..., end...]; DirectML takes the two halves
    // separately. auto_pad is the older spelling and only NOTSET is handled,
    // because SAME_UPPER/SAME_LOWER depend on the input size and would need
    // shape inference this translator does not do.
    const onnx::AttributeProto* autoPad = FindAttribute(node, "auto_pad");
    if (autoPad != nullptr && !autoPad->s().empty() && autoPad->s() != "NOTSET") {
        Fail("Conv node '" + node.name() + "' uses auto_pad=" + autoPad->s() +
             "; only explicit pads are supported");
    }

    std::vector<uint32_t> startPadding(spatial, 0);
    std::vector<uint32_t> endPadding(spatial, 0);
    const std::vector<uint32_t> pads = AttributeInts(node, "pads", {});
    if (!pads.empty()) {
        if (pads.size() != spatial * 2) {
            Fail("Conv node '" + node.name() + "' has " + std::to_string(pads.size()) +
                 " pads for " + std::to_string(spatial) + " spatial dimensions");
        }
        for (std::size_t i = 0; i < spatial; ++i) {
            startPadding[i] = pads[i];
            endPadding[i] = pads[i + spatial];
        }
    }

    const uint32_t groupCount = static_cast<uint32_t>(AttributeInt(node, "group", 1));

    return dml::ConvolutionBuilder(input, filter, bias)
        .Strides(strides)
        .Dilations(dilations)
        .StartPadding(startPadding)
        .EndPadding(endPadding)
        .GroupCount(groupCount)
        .Build();
}

dml::Expression TranslateGemm(const onnx::NodeProto& node, TranslationContext& ctx) {
    if (node.input_size() < 2) {
        Fail("Gemm node '" + node.name() + "' needs A and B");
    }
    auto a = ctx.Get(node.input(0));
    auto b = ctx.Get(node.input(1));

    dml::Optional<dml::Expression> c;
    if (node.input_size() >= 3 && !node.input(2).empty()) {
        c = ctx.Get(node.input(2));
    }

    const auto transform = [](int64_t flag) {
        return flag != 0 ? DML_MATRIX_TRANSFORM_TRANSPOSE : DML_MATRIX_TRANSFORM_NONE;
    };
    return dml::Gemm(a, b, c, transform(AttributeInt(node, "transA", 0)),
                     transform(AttributeInt(node, "transB", 0)),
                     AttributeFloat(node, "alpha", 1.0f),
                     AttributeFloat(node, "beta", 1.0f));
}

dml::Expression TranslateConcat(const onnx::NodeProto& node, TranslationContext& ctx) {
    const onnx::AttributeProto* axisAttr = FindAttribute(node, "axis");
    if (axisAttr == nullptr) {
        Fail("Concat node '" + node.name() + "' has no axis attribute");
    }
    std::vector<dml::Expression> inputs;
    inputs.reserve(node.input_size());
    for (const std::string& name : node.input()) {
        inputs.push_back(ctx.Get(name));
    }
    if (inputs.empty()) {
        Fail("Concat node '" + node.name() + "' has no inputs");
    }

    // ONNX allows a negative axis counting from the end; DirectML does not.
    int64_t axis = axisAttr->i();
    const int64_t rank = static_cast<int64_t>(inputs[0].GetOutputDesc().sizes.size());
    if (axis < 0) {
        axis += rank;
    }
    if (axis < 0 || axis >= rank) {
        Fail("Concat node '" + node.name() + "' has axis " + std::to_string(axisAttr->i()) +
             " for a rank-" + std::to_string(rank) + " input");
    }
    return dml::Join(inputs, static_cast<uint32_t>(axis));
}

}  // namespace

OnnxModel::OnnxModel() : m_asset(std::make_unique<Asset>()) {}
OnnxModel::~OnnxModel() = default;

std::vector<std::string> OnnxModel::SupportedOps() {
    return {"Add",  "Concat", "Conv",   "Gemm",  "Identity", "LeakyRelu",
            "MatMul", "Mul",  "Relu",   "Sigmoid", "Tanh"};
}

std::size_t OnnxModel::NodeCount() const {
    return static_cast<std::size_t>(m_asset->model.graph().node_size());
}

std::size_t OnnxModel::InitializerCount() const {
    return static_cast<std::size_t>(m_asset->model.graph().initializer_size());
}

std::unique_ptr<OnnxModel> OnnxModel::Load(const std::string& name,
                                           const std::filesystem::path& path,
                                           std::string* error) {
    const auto fail = [&](const std::string& message) -> std::unique_ptr<OnnxModel> {
        if (error != nullptr) {
            *error = message;
        }
        return nullptr;
    };

    GOOGLE_PROTOBUF_VERIFY_VERSION;

    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return fail("cannot open '" + path.string() + "'");
    }

    auto result = std::unique_ptr<OnnxModel>(new OnnxModel());
    if (!result->m_asset->model.ParseFromIstream(&file)) {
        return fail("'" + path.string() + "' is not a valid ONNX ModelProto");
    }
    result->m_name = name;

    const onnx::GraphProto& graph = result->m_asset->model.graph();
    if (graph.output_size() < 1) {
        return fail("ONNX graph has no outputs");
    }
    result->m_outputName = graph.output(0).name();

    // The model input is the graph input that is NOT an initializer. ONNX lists
    // initializers in graph.input() as well for older opsets, so filtering is
    // required rather than taking input(0).
    std::set<std::string> initializerNames;
    for (const onnx::TensorProto& tensor : graph.initializer()) {
        initializerNames.insert(tensor.name());
    }

    const onnx::ValueInfoProto* modelInput = nullptr;
    std::size_t realInputs = 0;
    for (const onnx::ValueInfoProto& value : graph.input()) {
        if (initializerNames.count(value.name()) != 0) {
            continue;
        }
        ++realInputs;
        if (modelInput == nullptr) {
            modelInput = &value;
        }
    }
    if (modelInput == nullptr) {
        return fail("ONNX graph has no non-initializer input to drive the model with");
    }
    if (realInputs > 1) {
        return fail("ONNX graph has " + std::to_string(realInputs) +
                    " inputs; ModelManager drives exactly one");
    }

    if (!modelInput->type().has_tensor_type()) {
        return fail("input '" + modelInput->name() + "' is not a tensor");
    }
    const onnx::TypeProto_Tensor& tensorType = modelInput->type().tensor_type();

    DML_TENSOR_DATA_TYPE dataType;
    if (!ToDmlDataType(tensorType.elem_type(), dataType)) {
        return fail("input '" + modelInput->name() + "' has element type " +
                    OnnxDataTypeName(tensorType.elem_type()) +
                    ", which DirectML does not support");
    }

    dml::TensorDesc::Dimensions sizes;
    for (const onnx::TensorShapeProto_Dimension& dim : tensorType.shape().dim()) {
        if (dim.has_dim_value() && dim.dim_value() > 0) {
            sizes.push_back(static_cast<uint32_t>(dim.dim_value()));
        } else {
            // A symbolic dimension (usually the batch) has no value in the
            // file. DirectML needs a concrete size, so it is pinned to 1 --
            // which is why this is reported rather than assumed silently.
            sizes.push_back(1);
        }
    }
    if (sizes.empty()) {
        return fail("input '" + modelInput->name() + "' has no shape");
    }

    result->m_inputName = modelInput->name();
    result->m_inputDesc = dml::TensorDesc(dataType, sizes);
    return result;
}

ModelGraph OnnxModel::Build(dml::Graph& graph, dml::Expression input) {
    const onnx::GraphProto& onnxGraph = m_asset->model.graph();

    ModelGraph result{input, {}};

    TranslationContext ctx;
    ctx.graph = &graph;
    ctx.onnxGraph = &onnxGraph;
    ctx.weights = &result.weights;
    ctx.tensors.emplace(m_inputName, input);
    for (const onnx::TensorProto& tensor : onnxGraph.initializer()) {
        ctx.initializers.emplace(tensor.name(), &tensor);
    }

    for (const onnx::NodeProto& node : onnxGraph.node()) {
        const std::string& op = node.op_type();
        if (node.output_size() < 1) {
            Fail("node '" + node.name() + "' (" + op + ") has no output");
        }

        dml::Expression produced = input;  // overwritten below in every branch

        if (op == "Conv") {
            produced = TranslateConv(node, ctx);
        } else if (op == "Gemm") {
            produced = TranslateGemm(node, ctx);
        } else if (op == "MatMul") {
            // MatMul is Gemm with no C, no transposes and unit scaling.
            produced = dml::Gemm(ctx.Get(node.input(0)), ctx.Get(node.input(1)));
        } else if (op == "Concat") {
            produced = TranslateConcat(node, ctx);
        } else if (op == "Add") {
            produced = ctx.Get(node.input(0)) + ctx.Get(node.input(1));
        } else if (op == "Mul") {
            produced = ctx.Get(node.input(0)) * ctx.Get(node.input(1));
        } else if (op == "Relu") {
            produced = dml::ActivationRelu(ctx.Get(node.input(0)));
        } else if (op == "LeakyRelu") {
            produced = dml::ActivationLeakyRelu(ctx.Get(node.input(0)),
                                                AttributeFloat(node, "alpha", 0.01f));
        } else if (op == "Sigmoid") {
            produced = dml::ActivationSigmoid(ctx.Get(node.input(0)));
        } else if (op == "Tanh") {
            produced = dml::ActivationTanh(ctx.Get(node.input(0)));
        } else if (op == "Identity") {
            produced = dml::Identity(ctx.Get(node.input(0)));
        } else {
            std::ostringstream message;
            message << "ONNX operator '" << op << "'";
            if (!node.name().empty()) {
                message << " (node '" << node.name() << "')";
            }
            message << " is not translated. Supported: ";
            const auto supported = SupportedOps();
            for (std::size_t i = 0; i < supported.size(); ++i) {
                message << (i == 0 ? "" : ", ") << supported[i];
            }
            Fail(message.str());
        }

        ctx.tensors[node.output(0)] = produced;
    }

    const auto output = ctx.tensors.find(m_outputName);
    if (output == ctx.tensors.end()) {
        Fail("graph output '" + m_outputName + "' was never produced");
    }
    result.output = output->second;
    return result;
}

}  // namespace NeuralModelIntegrateTestbed
