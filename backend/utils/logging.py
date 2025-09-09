
# Logging configuration for the Materials Property Predictor application.

import logging
import sys
from backend.config import settings


def setup_logging():
    # Configure application logging.
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format=settings.log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("app.log", mode="a")
        ]
    )
    
    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("streamlit").setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)


# Initialize logger
logger = setup_logging()