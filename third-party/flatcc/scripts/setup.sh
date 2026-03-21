#!/usr/bin/env bash

# This is intended for quickly developming with flatcc tools
# in a standalone directory

set -e

DIR=`pwd`
HERE=`dirname $0`
cd $HERE/..
ROOT=`pwd`

function usage() {
    echo "Usage: <flatcc-dir>/scripts/`basename $0` [options] <path>"
    echo ""
    echo "Options:"
    echo "      -g | --gitignore    : create/update .gitignore file"
    echo "      -b | --build        : build flatcc (otherwise is must have been)"
    echo "      -x | --example      : copy example source and schema"
    echo "      -s | --script       : copy generic build script"
    echo "      -a | --all          : all of the above"
    echo "      -h | --help"
    echo ""
    echo "Sets up a client project for use with flatcc."
    echo ""
    echo "Links flatcc into bin, lib, and include directories and optionally"
    echo "starts a build first. Optionally creates or updates a .gitignore file"
    echo "and a generic build script, and a sample project."
    echo "Also adds an empty generated directory for 'flatcc -o generated',"
    echo "'cc -I generated', and for git to ignore. 'build' directory"
    echo "will be ignored if '-b' is selected."
    echo ""
    echo "When using the build script (-s), place source and schema files in 'src'."
    echo "It is only meant for sharing small examples."
    echo ""
    echo "The flatcc project must be the parent of the path to this script."
    exit 1
}

while [  $# -gt 0 ]; do
  case "$1" in

  # Standard help option.
  -h|-\?|-help|--help|--doc*) usage ;;
  -g|--gitignore) G=1 ;;
  -b|--build) B=1 ;;
  -s|--script) S=1 ;;
  -x|--example) X=1 ;;
  -a|--all) G=1; B=1; S=1; X=1 ;;

  -*) echo "Unknown option \"$1\""; usage ;;
  *)  break ;;           # unforced  end of user options
  esac
  shift   # next option
done

if [[ -z "$1" ]]; then
   echo "Please specify a path"
   usage
fi

echo "Building flatcc libraries and tool"

if [[ ! -d "$ROOT/include/flatcc" ]]; then
    echo "script not located in flatcc project"
fi

if [[ -n "$B" ]]; then
    $ROOT/scripts/build.sh
fi

echo "Linking flatcc tool and library into $1"

mkdir -p $DIR/$1
cd $DIR/$1

if [[ -n "$S" ]]; then
    echo "Copying build script"
    mkdir -p scripts
    mkdir -p src
    cp $ROOT/scripts/_user_build.in scripts/build.sh
    chmod +x scripts/build.sh
fi

if [[ -n "$X" ]]; then
    echo "Copying monster sample project"
    mkdir -p src
    cp $ROOT/samples/monster/monster.{c,fbs} src
fi

mkdir -p lib
mkdir -p bin
mkdir -p include

ln -sf $ROOT/bin/flatcc bin/
ln -sf $ROOT/lib/libflatcc.a lib/
ln -sf $ROOT/lib/libflatccrt.a lib/
ln -sf $ROOT/lib/libflatcc_d.a lib/
ln -sf $ROOT/lib/libflatccrt_d.a lib/
ln -sf $ROOT/include/flatcc include/

if [[ -n "$G" ]]; then
    echo "Updating .gitignore"
    touch .gitignore
    grep -q '^bin/flatcc*' .gitignore || echo 'bin/flatcc*' >> .gitignore
    grep -q '^lib/libflatcc*.a' .gitignore || echo 'lib/libflatcc*.a' >> .gitignore
    grep -q '^include/flatcc' .gitignore || echo 'include/flatcc' >> .gitignore
    grep -q '^generated/' .gitignore || echo 'generated/' >> .gitignore
    if [[ -n "$S" ]]; then
        grep -q '^build/' .gitignore || echo 'build/' >> .gitignore
    fi
fi
