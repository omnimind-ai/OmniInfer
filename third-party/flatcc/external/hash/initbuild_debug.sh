#!/usr/bin/env bash

cd `dirname $0`
mkdir -p "build/debug"
cd build/debug && cmake -GNinja ../.. -DCMAKE_BUILD_TYPE=Debug && ninja
