"""Event extraction from synthetic SAM tracks: a hand carries a box onto a plate."""

import numpy as np

from reactsim.imitation.demonstration import Demonstration
from reactsim.imitation.events import extract_events
from reactsim.perception.video import Track

H, W, N = 120, 200, 30


def _box(cx, cy, r):
    m = np.zeros((H, W), bool)
    m[max(cy - r, 0):cy + r, max(cx - r, 0):cx + r] = True
    return m


def _scene():
    hand, cup, plate = Track("hand"), Track("cup"), Track("plate")
    for i in range(N):
        plate.masks[i] = _box(150, 80, 25)
        x = 50 if i < 10 else 150 if i >= 20 else 50 + (i - 10) * 10   # carried in frames 10-19
        cup.masks[i] = _box(x, 70, 8)
        if 6 <= i < 23:
            hand.masks[i] = _box(x - 12, 70, 6)
    return hand, cup, plate


def test_extracts_moved_object_destination_and_timing():
    hand, cup, plate = _scene()
    ev = extract_events([hand], [cup, plate], N, W)
    assert ev.target is cup and ev.reference is plate
    assert ev.grasp_frame < 11 and ev.grasp_frame in ev.contact_frames
    assert 19 <= ev.release_frame <= 21


def test_nothing_moved_is_an_error():
    hand, _, plate = _scene()
    still = Track("cup", masks={i: _box(50, 70, 8) for i in range(N)})
    try:
        extract_events([hand], [still, plate], N, W)
    except ValueError as e:
        assert "nothing moved" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_instruction_carries_human_grasp_as_hint():
    d = Demonstration("v.mp4", "paper cup", "paper plate", "cup", "plate", "angled",
                      "Fingers over the rim.", 3, 9)
    text = d.instruction()
    assert text.startswith("Put the cup on the plate.")
    assert "an angled grasp" in text and "hint, not a rule" in text
