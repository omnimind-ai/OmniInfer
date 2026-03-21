#!/bin/sh

cd $(dirname $0)/..
builddir=build/Debug
bfbsdir=$builddir/test/reflection_test/generated/

if [ ! -e $bfbsdir/monster_test.bfbs ]; then
    scripts/test.sh
fi

$builddir/samples/reflection/bfbs2json_d \
    $bfbsdir/monster_test.bfbs > $bfbsdir/monster_test_bfbs.json

cat $bfbsdir/monster_test_bfbs.json \
    | python -m json.tool
