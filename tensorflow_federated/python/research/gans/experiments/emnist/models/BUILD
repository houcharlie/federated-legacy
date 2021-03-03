package(default_visibility = ["//tensorflow_federated/python/research/gans/experiments/emnist:__subpackages__"])

licenses(["notice"])

py_library(
    name = "convolutional_gan_networks",
    srcs = ["convolutional_gan_networks.py"],
    deps = ["custom_layers"],
    srcs_version = "PY3",
)
py_library(
    name = "custom_layers",
    srcs = ["custom_layers.py"],
    deps = ["scale"],
    srcs_version = "PY3",
)
py_library(
    name = "scale",
    srcs = ["scale.py"],
    srcs_version = "PY3",
)