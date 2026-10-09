// Proves the vcpkg onnx dependency is wired up: that the protobuf-generated
// headers compile, that the schema links (onnx.lib + onnx_proto.lib +
// libprotobuf), and that a ModelProto can be built and round-tripped through
// its serialized form. No DirectML and no device involved.
#include <onnx/onnx_pb.h>

#include <cstdio>
#include <string>

int main() {
    // Protobuf requires its version check before any generated class is used.
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    onnx::ModelProto model;
    model.set_ir_version(onnx::IR_VERSION);
    model.set_producer_name("ptflio onnx-smoke");
    auto* opset = model.add_opset_import();
    opset->set_domain("");
    opset->set_version(13);

    onnx::GraphProto* graph = model.mutable_graph();
    graph->set_name("Identity");

    onnx::NodeProto* node = graph->add_node();
    node->set_op_type("Relu");
    node->add_input("x");
    node->add_output("y");

    std::string bytes;
    if (!model.SerializeToString(&bytes)) {
        std::printf("FAIL: SerializeToString\n");
        return 1;
    }

    onnx::ModelProto parsed;
    if (!parsed.ParseFromString(bytes)) {
        std::printf("FAIL: ParseFromString\n");
        return 1;
    }

    const bool ok = parsed.graph().node_size() == 1 &&
                    parsed.graph().node(0).op_type() == "Relu" &&
                    parsed.producer_name() == "ptflio onnx-smoke";
    std::printf("serialized %zu bytes, round-trip %s\n", bytes.size(), ok ? "ok" : "MISMATCH");
    std::printf("onnx IR_VERSION %lld, protobuf %d\n",
                static_cast<long long>(onnx::IR_VERSION), GOOGLE_PROTOBUF_VERSION);
    google::protobuf::ShutdownProtobufLibrary();
    return ok ? 0 : 1;
}
