#!/bin/sh

set -e

cd $(dirname $0)
mkdir -p build

CC=cc

$CC -g -Wall -Wextra $INCLUDE -I.. grisu3_test.c -lm -o build/grisu3_test_d
$CC -DNDEBUG -Wall -Wextra -O2 $INCLUDE -I.. grisu3_test.c -lm -o build/grisu3_test
echo "DEBUG:"
build/grisu3_test_d
echo "OPTIMIZED:"
build/grisu3_test

echo "running double conversion tests"
./test_dblcnv.sh
