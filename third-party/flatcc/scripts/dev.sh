#!/usr/bin/env sh

set -e

HERE=`dirname $0`
cd $HERE/..
ROOT=`pwd`

${ROOT}/scripts/test.sh --debug --no-clean
