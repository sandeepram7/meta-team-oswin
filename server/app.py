from openenv.core.env_server import create_fastapi_app
from .environment import DataCurationEnv

# Create the FastAPI app with our environment
app = create_fastapi_app(DataCurationEnv)
