"""Graph input/output names, dtypes and shapes of an .onnx file.

A minimal protobuf wire-format scanner, so preparing assets needs nothing but
the Python standard library -- the `onnx` package is a heavy dependency to
require just to read a handful of tensor shapes.

Only the fields needed are walked:
    ModelProto.graph       = 7  (message)
    GraphProto.initializer = 5  (message)  -- to exclude weights from `input`
    GraphProto.input       = 11 (message)
    GraphProto.output      = 12 (message)
    ValueInfoProto.name    = 1  (string), .type = 2 (message)
    TypeProto.tensor_type  = 1  -> .elem_type = 1 (varint), .shape = 2 (message)
    TensorShapeProto.dim   = 1  -> .dim_value = 1 (varint), .dim_param = 2 (string)
    TensorProto.name       = 8  (string)

A dimension is reported as an int when the model fixes it, or as the symbolic
name string when it is dynamic.
"""

DATA_TYPES = {
    1: "float32", 2: "uint8", 3: "int8", 4: "uint16", 5: "int16", 6: "int32",
    7: "int64", 9: "bool", 10: "float16", 11: "float64", 12: "uint32",
    13: "uint64", 16: "bfloat16",
}


def _varint(buf, i):
    result = shift = 0
    while True:
        byte = buf[i]
        i += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, i
        shift += 7


def _fields(buf, start=0, end=None):
    """Yield (field_number, wire_type, payload) for one protobuf message."""
    end = len(buf) if end is None else end
    i = start
    while i < end:
        key, i = _varint(buf, i)
        number, wire = key >> 3, key & 7
        if wire == 0:
            value, i = _varint(buf, i)
            yield number, wire, value
        elif wire == 2:
            length, i = _varint(buf, i)
            yield number, wire, buf[i:i + length]
            i += length
        elif wire == 5:
            yield number, wire, buf[i:i + 4]
            i += 4
        elif wire == 1:
            yield number, wire, buf[i:i + 8]
            i += 8
        else:
            raise ValueError("unsupported protobuf wire type %d" % wire)


def _value_info(buf):
    name, elem_type, dims = "", None, []
    for number, wire, value in _fields(buf):
        if number == 1 and wire == 2:
            name = value.decode("utf-8", "replace")
        elif number == 2 and wire == 2:
            for n2, w2, v2 in _fields(value):
                if n2 != 1 or w2 != 2:  # tensor_type only
                    continue
                for n3, w3, v3 in _fields(v2):
                    if n3 == 1 and w3 == 0:
                        elem_type = v3
                    elif n3 == 2 and w3 == 2:
                        for n4, w4, v4 in _fields(v3):
                            if n4 != 1 or w4 != 2:
                                continue
                            dim = None
                            for n5, w5, v5 in _fields(v4):
                                if n5 == 1 and w5 == 0:
                                    dim = v5
                                elif n5 == 2 and w5 == 2:
                                    dim = v5.decode("utf-8", "replace")
                            dims.append(dim)
    return {"name": name, "dtype": DATA_TYPES.get(elem_type, str(elem_type)),
            "shape": dims}


def signature(path):
    """{"inputs": [...], "outputs": [...]} with initializers left out of inputs."""
    with open(path, "rb") as handle:
        buf = handle.read()
    graph = None
    for number, wire, value in _fields(buf):
        if number == 7 and wire == 2:
            graph = value
    if graph is None:
        raise ValueError("%s has no GraphProto" % path)

    inputs, outputs, initializers = [], [], set()
    for number, wire, value in _fields(graph):
        if number == 11 and wire == 2:
            inputs.append(_value_info(value))
        elif number == 12 and wire == 2:
            outputs.append(_value_info(value))
        elif number == 5 and wire == 2:
            for n2, w2, v2 in _fields(value):
                if n2 == 8 and w2 == 2:
                    initializers.add(v2.decode("utf-8", "replace"))
    return {"inputs": [i for i in inputs if i["name"] not in initializers],
            "outputs": outputs}


def _format(entry):
    shape = ",".join(str(d) for d in entry["shape"])
    return "%-14s %-9s [%s]" % (entry["name"], entry["dtype"], shape)


if __name__ == "__main__":
    import sys

    for path in sys.argv[1:]:
        sig = signature(path)
        print("===", path)
        for entry in sig["inputs"]:
            print("  IN  " + _format(entry))
        for entry in sig["outputs"]:
            print("  OUT " + _format(entry))
