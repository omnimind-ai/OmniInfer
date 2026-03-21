#!/usr/bin/env bash

cd `dirname $0`
mkdir -p "build/release"
cd build/release && cmake -GNinja ../.. -DCMAKE_BUILD_TYPE=Release && ninja
