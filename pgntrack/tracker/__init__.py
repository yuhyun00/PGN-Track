"""ByteTrack association (McByte configuration) with a pluggable motion stage and plugin hooks."""
from .track import Track, TrackState
from .tracker import FrameEvents, Tracker

__all__ = ['FrameEvents', 'Track', 'TrackState', 'Tracker']
