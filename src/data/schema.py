"""Shared sample schema; -1 is the only missing-label sentinel."""

from typing import Any, TypedDict

DROWSINESS = ("Alert", "Eyes closed/sleeping", "Yawning", "Head nodding")
DISTRACTION = (
    "Safe driving",
    "Mobile phone use",
    "Eating or drinking",
    "Other object",
    "Gaze off road",
    "Talking to passenger",
)


class Sample(TypedDict):
    image: Any
    label_distraction: int
    label_drowsiness: int
    subject_id: str
    dataset: str
    metadata: dict


def validate_labels(distraction, drowsiness):
    if distraction not in range(-1, len(DISTRACTION)):
        raise ValueError("Invalid distraction label")
    if drowsiness not in range(-1, len(DROWSINESS)):
        raise ValueError("Invalid drowsiness label")
