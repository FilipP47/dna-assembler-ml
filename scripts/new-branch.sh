#!/bin/bash
# Użycie: bash scripts/new-branch.sh feat 2 login-form
TYPE=$1
ISSUE=$2
DESC=$3

if [ -z "$TYPE" ] || [ -z "$ISSUE" ] || [ -z "$DESC" ]; then
    echo "Użycie: bash scripts/new-branch.sh <typ> <numer> <opis>"
    exit 1
fi

BRANCH_NAME="${TYPE}/${ISSUE}-${DESC}"
git checkout develop
git pull origin develop
git checkout -b "$BRANCH_NAME"
echo "Stworzono branch: $BRANCH_NAME"