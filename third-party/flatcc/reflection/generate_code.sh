#!/usr/bin/env bash

# Regnerate reflection API
#
# The output should be checked in with the project since it is
# a bootstrapping process.

cd `dirname $0`
../scripts/build.sh
RPATH=../include/flatcc/reflection
mkdir -p ${RPATH}
../bin/flatcc -a -o ../include/flatcc/reflection reflection.fbs
cp README.in ${RPATH}/README
