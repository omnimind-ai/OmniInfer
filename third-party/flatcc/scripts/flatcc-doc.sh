#!/bin/sh

HOME=$(dirname $0)/..
SCHEMA=${SCHEMA:-$1}
PREFIX=${PREFIX:-$2}
OUTDIR=${OUTDIR:-$3}
OUTDIR=${OUTDIR:-'.'}
INCLUDE=${INCLUDE:-$HOME/include}
FLATCC=${FLATCC:-$HOME/bin/flatcc}

if [ "x$SCHEMA" = "x" ]; then
    echo "Missing schema arg"
    echo "usage: $(basename $0) schema-file name-prefix [outdir]"
    exit 1
fi

if [ "x$PREFIX" = "x" ]; then
    echo "Missing prefix arg"
    echo "usage: $(basename $0) schema-file name-prefix [outdir]"
    exit 1
fi

echo "flatcc doc for schema: '$SCHEMA' with name prefix: '$PREFIX'"

echo "generating $OUTDIR/$PREFIX.doc"

$FLATCC $SCHEMA -a --json --stdout | \
    clang - -E -DNDEBUG -I $INCLUDE | \
    clang-format -style="WebKit" | \
    grep "^static.* $PREFIX\w*(" | \
    cut -f 1 -d '{' | \
    grep -v deprecated | \
    grep -v ");$" | \
    sed 's/__tmp//g' | \
    sed 's/)/);/g' \
    > $OUTDIR/$PREFIX.doc
