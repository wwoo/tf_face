# PubFig bazel build file

package(default_visibility = [":internal"])

exports_files(["LICENSE"])

package_group(
    name = "internal",
    packages = ["//pubfig/..."],
)

py_binary(
    name = "export",
    srcs = [
        "export.py",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(
        [
            "**/*.py",
            "BUILD",
        ],
    ),
)
