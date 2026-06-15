#!/bin/sh
# =============================================================================
# pre-push
# Blokuje push jesli nazwa brancha nie pasuje do konwencji
# Format: type/numer-opis  np. feat/12-login-form
# Galęzie chronione (zawsze przepuszczane): main, develop
# =============================================================================

BRANCH=$(git symbolic-ref --short HEAD 2>/dev/null)

# Przepusc galezie chronione
if [ "$BRANCH" = "main" ] || [ "$BRANCH" = "develop" ]; then
  exit 0
fi

# Walidacja formatu
# Dozwolone: feat/12-opis, bug/7-opis, docs/15-opis, chore/3-opis
PATTERN='^(feat|bug|docs|chore)/[0-9]+-[a-z0-9-]+$'

if ! echo "$BRANCH" | grep -qE "$PATTERN"; then
  echo ""
  echo "ERROR: Bledna nazwa brancha: '$BRANCH'"
  echo ""
  echo "  Wymagany format:  type/numer-krotki-opis"
  echo "  Przyklady:"
  echo "    feat/12-login-form"
  echo "    bug/7-email-validation"
  echo "    docs/15-readme-api"
  echo "    chore/3-update-deps"
  echo ""
  echo "  Dozwolone typy: feat, bug, docs, chore"
  echo ""
  exit 1
fi
