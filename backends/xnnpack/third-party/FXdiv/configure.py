#!/usr/bin/env python


import confu
parser = confu.standard_parser("FXdiv configuration script")


def main(args):
    options = parser.parse_args(args)
    build = confu.Build.from_options(options)

    build.export_cpath("include", ["fxdiv.h"])

    with build.options(source_dir="test", deps=build.deps.googletest):
        build.unittest("multiply-high-test", build.cxx("multiply-high.cc"))
        build.unittest("quotient-test", build.cxx("quotient.cc"))

    with build.options(source_dir="bench", deps=build.deps.googlebenchmark):
        build.benchmark("init-bench", build.cxx("init.cc"))
        build.benchmark("multiply-bench", build.cxx("multiply.cc"))
        build.benchmark("divide-bench", build.cxx("divide.cc"))
        build.benchmark("quotient-bench", build.cxx("quotient.cc"))
        build.benchmark("round-down-bench", build.cxx("round-down.cc"))

    return build


if __name__ == "__main__":
    import sys
    main(sys.argv[1:]).generate()
