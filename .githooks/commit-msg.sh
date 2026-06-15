#!/bin/sh
# =============================================================================
# commit-msg
# 1. Jesli brak #N w commicie — dopisuje go automatycznie z nazwy brancha
# 2. Jesli brak typu — dopisuje DEFAULT_TYPE (ustaw ponizej)
#    Aby wylaczyc domyslny typ: ustaw DEFAULT_TYPE=""
# 3. Waliduje format: type(#N): opis
# =============================================================================

DEFAULT_TYPE="feat"

# =============================================================================

MSG=$(cat "$1")
BRANCH=$(git symbolic-ref --short HEAD 2>/dev/null)
NUMBER=$(echo "$BRANCH" | grep -oE '[0-9]+' | head -1)

# Sprawdz czy commit ma juz prawidlowy typ
PATTERN='^(feat|bug|docs|chore)(\(#[0-9]+\))?: .+'
HAS_TYPE=$(echo "$MSG" | grep -E "$PATTERN")

if [ -n "$HAS_TYPE" ]; then
  # Format poprawny — tylko dopisz #N jesli brak
  if [ -n "$NUMBER" ] && ! echo "$MSG" | grep -q "#$NUMBER"; then
    MSG="$MSG (#$NUMBER)"
  fi
else
  # Brak typu — uzyj DEFAULT_TYPE lub blad
  if [ -z "$DEFAULT_TYPE" ]; then
    echo ""
    echo "ERROR: Bledny format commita i brak domyslnego typu."
    echo ""
    echo "  Twoj commit:  $MSG"
    echo ""
    echo "  Wymagany format:  type(#N): krotki opis"
    echo "  Przyklady:"
    echo "    feat(#12): add login form"
    echo "    bug(#7): fix email validation"
    echo "    docs(#15): update readme"
    echo "    chore(#3): update dependencies"
    echo ""
    exit 1
  fi

  # Zbuduj wiadomosc od nowa: type(#N): oryginalny opis
  if [ -n "$NUMBER" ]; then
    MSG="$DEFAULT_TYPE(#$NUMBER): $MSG"
  else
    MSG="$DEFAULT_TYPE: $MSG"
  fi
fi

echo "$MSG" > "$1"
