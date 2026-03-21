#!/usr/bin/env bash

set -e
cd `dirname $0`/../..
ROOT=`pwd`
TMP=${ROOT}/build/tmp/samples/reflection

CC=${CC:-cc}
${ROOT}/scripts/build.sh
mkdir -p ${TMP}
rm -rf ${TMP}/*
#bin/flatcc --schema --schema-length=yes -o ${TMP} test/monster_test/monster_test.fbs
bin/flatcc --schema -o ${TMP} test/monster_test/monster_test.fbs

cp samples/reflection/*.c ${TMP}
cd ${TMP}
# We don't need debug version, but it is always useful to have if we get
# assertions in the interface code.
$CC -g -I ${ROOT}/include bfbs2json.c -o bfbs2jsond
$CC -O3 -DNDEBUG -I ${ROOT}/include bfbs2json.c -o bfbs2json
cp bfbs2json ${ROOT}/bin/bfbs2json
echo "generating example json output from monster_test.fbs schema ..."
${ROOT}/bin/bfbs2json ${TMP}/monster_test.bfbs > monster_test_schema.json
cat monster_test_schema.json | python -m json.tool > pretty_monster_test_schema.json
echo "test json file located in ${TMP}/monster_test_schema.json"
echo "pretty printed file located in ${TMP}/pretty_monster_test_schema.json"
echo "bfbs2json tool placed in ${ROOT}/bin/bfbs2json"
