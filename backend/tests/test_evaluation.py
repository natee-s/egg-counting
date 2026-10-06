"""A merged prediction must not pass an audit for two physical eggs."""
import numpy as np
from backend.evaluate_models import match_references


def test_merged_instance_cannot_explain_two_eggs():
    points = [[20, 20], [40, 20]]
    boxes = np.array([[10, 10, 50, 30]])
    assert len(match_references(points, boxes, [])) == 1


def test_box_covering_point_but_wrong_mask_is_a_miss():
    boxes = np.array([[0, 0, 50, 50]])
    polygon = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], np.float32)
    assert match_references([[30, 30]], boxes, [polygon]) == set()


def test_one_to_one_matching_preserves_two_adjacent_instances():
    boxes = np.array([[0, 0, 50, 30], [10, 10, 30, 30]])
    # The first reference fits both predictions; the second fits only one.
    # Reassignment is necessary to avoid incorrectly reporting a missed egg.
    assert match_references([[20, 20], [40, 20]], boxes, []) == {0, 1}
