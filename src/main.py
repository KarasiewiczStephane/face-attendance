"""Entry point for the face attendance system API server."""

import uvicorn

from src.utils.config import get_settings


def main() -> None:
    """Start the uvicorn server with configuration from config.yaml."""
    settings = get_settings()
    config = settings.load_config()
    api_cfg = config.get("api", {})

    uvicorn.run(
        "src.api.app:app",
        host=api_cfg.get("host", "0.0.0.0"),
        port=api_cfg.get("port", 8000),
        reload=api_cfg.get("reload", False),
    )


if __name__ == "__main__":
    main()
