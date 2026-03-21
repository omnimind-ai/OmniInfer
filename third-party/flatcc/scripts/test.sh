#!/usr/bin/env sh

set -e

HERE=`dirname $0`
cd $HERE/..
ROOT=`pwd`

DBGDIR=$ROOT/build/Debug
RELDIR=$ROOT/build/Release

if [ "$1" = "--debug" ]; then
    DEBUG=$1
    echo "running debug build"
    shift
fi

if [ "$1" != "--no-clean" ]; then
    echo "cleaning build before tests ..."
    $ROOT/scripts/cleanall.sh
else
    shift
fi

echo "building before tests ..."
$ROOT/scripts/build.sh $DEBUG

echo "running test in debug build ..."
cd $DBGDIR && ctest $ROOT

if [ "$DEBUG" != "--debug" ]; then
echo "running test in release build ..."
cd $RELDIR && ctest $ROOT
echo "TEST PASSED"
else
    echo "DEBUG TEST PASSED"
fi

