"""Per-env converters: env-specific observation → shared object-list Frame."""
from .arc_agi_3 import arc_agi_3_to_frame
from .minigrid import minigrid_to_frame

__all__ = ["arc_agi_3_to_frame", "minigrid_to_frame"]
