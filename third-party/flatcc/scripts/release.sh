#!/usr/bin/env bash

VER=`git describe --tags`

echo "archiving tagged version ${VER} to release folder"

cd `dirname $0`/..
mkdir -p release
git archive --format=tar.gz --prefix=flatcc-$VER/ v$VER >release/flatcc-$VER.tar.gz
