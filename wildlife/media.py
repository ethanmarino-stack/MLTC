"""Short-lived video capabilities for the current Demo session, not file browsing."""
from pathlib import Path
import secrets


class SessionMedia:
    """Register only a resolved, selected video after OpenCV has validated it."""

    def __init__(self):
        self.references: dict[str, tuple[str, Path]] = {}

    def register(self, camera_id: str, path: Path) -> str:
        token = secrets.token_urlsafe(32)
        self.references[token] = (camera_id, path.resolve())
        return "/api/media/" + token

    def get(self, token: str) -> Path | None:
        reference = self.references.get(token)
        if reference is None:
            return None
        path = reference[1]
        # Reject a path replaced by a symlink/junction after selection. The URL
        # never accepts a filename, directory, or caller-supplied filesystem path.
        try:
            return path if path.is_file() and path.resolve() == path else None
        except (OSError, RuntimeError):
            return None

    def remove_camera(self, camera_id: str):
        for token, (owner, _) in list(self.references.items()):
            if owner == camera_id:
                self.references.pop(token, None)

    def clear(self):
        self.references.clear()
