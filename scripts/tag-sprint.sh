#!/bin/bash
# Użycie: bash scripts/tag-sprint.sh 3
SPRINT_NUM=$1

if [ -z "$SPRINT_NUM" ]; then
    echo "Podaj numer sprintu!"
    exit 1
fi

TAG="sprint-${SPRINT_NUM}"
git tag "$TAG"
echo "Stworzono tag: $TAG"