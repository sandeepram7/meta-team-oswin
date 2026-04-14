"""
FastAPI entry point for the Meta Data Curation Lab environment.
"""

import uvicorn
from openenv.core.env_server import create_fastapi_app

from .environment import DataCurationEnv
from .models import DataCleanAction, DataCleanObservation

app = create_fastapi_app(DataCurationEnv, DataCleanAction, DataCleanObservation)


def main():
    """Entry point for the server script."""
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
