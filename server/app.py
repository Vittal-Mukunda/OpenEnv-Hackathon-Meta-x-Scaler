"""OpenEnv server entry point.

Re-exports the FastAPI application so that `openenv validate` can discover
the server at the expected ``server/app.py`` location.
"""

from __future__ import annotations

import sys
import os

# Add parent dir so that top-level modules (environment, models, graders) are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import app  # noqa: E402, F401

def main() -> None:
    """Run the server with uvicorn."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
