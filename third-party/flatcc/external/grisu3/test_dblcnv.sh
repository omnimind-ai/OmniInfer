#!/bin/sh

set -e

cd $(dirname $0)
mkdir -p build

CC=cc

$CC -g -Wall -Wextra $INCLUDE -I.. grisu3_test_dblcnv.c -o build/grisu3_test_dblcnv_d
$CC -DNDEBUG -Wall -Wextra -O2  $INCLUDE -I.. grisu3_test_dblcnv.c -o build/grisu3_test_dblcnv
echo "DEBUG:"
build/grisu3_test_dblcnv_d
echo "OPTIMIZED:"
build/grisu3_test_dblcnv
