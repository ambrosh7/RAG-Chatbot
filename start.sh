#!/bin/sh
# Railway sets PORT. Expand it here so uvicorn never receives the literal "$PORT".
set -eu
PORT_VALUE=$(python -c 'import os, sys
value = os.environ.get("PORT", "8000").strip()
if not value.isdigit():
    raise SystemExit("PORT must be a number")
print(value)')
exec uvicorn src.api.main:app --host 0.0.0.0 --port "${PORT_VALUE}"
