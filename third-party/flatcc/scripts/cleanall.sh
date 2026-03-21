#!/bin/sh

set -e

echo "removing build products"

cd `dirname $0`/..

rm -rf build
rm -rf release
rm -f bin/flatcc*
rm -f bin/bfbs2json*
rm -f lib/libflatcc*
if [ -d bin ]; then
    rmdir bin
fi
if [ -d lib ]; then
    rmdir lib
fi

