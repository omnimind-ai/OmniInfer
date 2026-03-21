#!/usr/bin/env bash

set -e

cd `dirname $0`/..
test/benchmark/benchall.sh
