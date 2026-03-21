#!/usr/bin/env bash

set -e
cd `dirname $0`/../..
ROOT=`pwd`
NAME=monster
TMP=${ROOT}/build/tmp/samples/${NAME}
EX=${ROOT}/samples/${NAME}

CC=${CC:-cc}
CFLAGS_DEBUG="-g -I ${ROOT}/include"
CFLAGS_RELEASE="-O3 -DNDEBUG -I ${ROOT}/include"
${ROOT}/scripts/build.sh
mkdir -p ${TMP}
rm -rf ${TMP}/*
bin/flatcc -a -o ${TMP} ${EX}/${NAME}.fbs

cp ${EX}/*.c ${TMP}
cd ${TMP}

echo "building $NAME example (debug)"
$CC $CFLAGS_DEBUG ${NAME}.c ${ROOT}/lib/libflatccrt_d.a -o ${NAME}_d
echo "building $NAME example (release)"
$CC $CFLAGS_RELEASE ${NAME}.c ${ROOT}/lib/libflatccrt.a -o ${NAME}

echo "running $NAME example (debug)"
./${NAME}_d
