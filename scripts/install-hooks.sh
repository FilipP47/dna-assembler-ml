#!/bin/bash
# Plik: scripts/install-hooks.sh
git config core.hooksPath .githooks
chmod +x .githooks/pre-commit
echo "Git hooks zainstalowane."