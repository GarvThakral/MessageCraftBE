try:
    from main import app
except ModuleNotFoundError:  # pragma: no cover
    from backend.main import app
