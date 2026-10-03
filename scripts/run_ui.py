"""Launch the chat page against the local API.

Run from the repo root:

    python scripts/run_ui.py

This sets ``CHAT_API_BASE_URL`` to the local API when it is not already
set, so the page keeps calling ``POST /chat``. Community Cloud runs
``src/ui/app.py`` directly and leaves that variable unset, which answers
in-process.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.ui.app import LOCAL_API_BASE_URL


def main() -> None:
    from streamlit.web import cli as stcli

    os.environ.setdefault("CHAT_API_BASE_URL", LOCAL_API_BASE_URL)
    app = Path(__file__).resolve().parent.parent / "src" / "ui" / "app.py"
    sys.argv = ["streamlit", "run", str(app), *sys.argv[1:]]
    raise SystemExit(stcli.main())


if __name__ == "__main__":
    main()
