from openenv.core.env_server import create_fastapi_app
from .environment import DataCurationEnv
from .models import DataCleanAction, DataCleanObservation

# Create the FastAPI app with our environment and its types
app = create_fastapi_app(
    DataCurationEnv,
    DataCleanAction,
    DataCleanObservation
)
