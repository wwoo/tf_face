# PubFig bazel build file

package(
    default_visibility = ["//tensorflow_serving:internal"],
    features = ["no_layering_check"],
)

load("//tensorflow_serving:serving.bzl", "serving_proto_library")

py_binary(
    name = "predict_serving",
    srcs = [
        "predict_serving.py",
    ],
    deps = [
        "//tensorflow_serving/apis:predict_proto_py_pb2",
        "//tensorflow_serving/apis:prediction_service_proto_py_pb2",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
    imports = [
        "lib",
        "templates"
    ],
    data = glob(["templates/**", "lib/**", "*.json"])
)
