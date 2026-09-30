from src.detection import (
    box_area,
    calculate_iou,
    evaluate_detections,
    precision,
    recall,
    average_precision,
    build_tp_fp_outcomes,
    mean_average_precision,
    calculate_map,
    xyxy_to_yolo
)


def test_box_area():
    """Verify that bounding-box area is calculated correctly."""
    assert box_area(0, 0, 10, 10) == 100


def test_iou_no_overlap():
    """Verify that non-overlapping boxes have an IoU of zero."""
    box1 = (0, 0, 10, 10)
    box2 = (20, 20, 30, 30)

    assert calculate_iou(box1, box2) == 0


def test_iou_identical_boxes():
    """Verify that identical boxes have an IoU of one."""
    box = (0, 0, 10, 10)

    assert calculate_iou(box, box) == 1


def test_evaluate_detections():
    """Verify TP, FP, and FN counts for a mixed set of detections."""
    predictions = [
        {
            "label": "flour",
            "box": (100, 100, 300, 300),
        },
        {
            "label": "sugar",
            "box": (400, 100, 600, 300),
        },
        {
            "label": "banana",
            "box": (700, 100, 800, 200),
        },
    ]

    ground_truths = [
        {
            "label": "flour",
            "box": (110, 110, 290, 290),
        },
        {
            "label": "sugar",
            "box": (410, 110, 590, 290),
        },
        {
            "label": "milk",
            "box": (900, 100, 1000, 300),
        },
    ]

    tp, fp, fn = evaluate_detections(
        predictions,
        ground_truths,
        iou_threshold=0.5,
    )

    assert tp == 2
    assert fp == 1
    assert fn == 1


def test_precision():
    """Verify precision is calculated from true and false positives."""
    assert precision(8, 2) == 0.8


def test_recall():
    """Verify recall is calculated from true positives and false negatives."""
    assert recall(8, 4) == 8 / 12


def test_average_precision():
    """Verify simplified AP for a ranked sequence of TP/FP outcomes."""
    outcomes = [1, 0, 1, 1, 0]

    ap = average_precision(
        outcomes,
        num_ground_truths=3,
    )

    assert round(ap, 4) == 0.8056


def test_build_tp_fp_outcomes():
    """Verify predictions are converted into confidence-ranked TP/FP outcomes."""
    predictions = [
        {
            "label": "flour",
            "confidence": 0.95,
            "box": (100, 100, 300, 300),
        },
        {
            "label": "banana",
            "confidence": 0.90,
            "box": (700, 100, 800, 200),
        },
        {
            "label": "sugar",
            "confidence": 0.80,
            "box": (400, 100, 600, 300),
        },
    ]

    ground_truths = [
        {
            "label": "flour",
            "box": (110, 110, 290, 290),
        },
        {
            "label": "sugar",
            "box": (410, 110, 590, 290),
        },
    ]

    outcomes = build_tp_fp_outcomes(
        predictions,
        ground_truths,
        iou_threshold=0.5,
    )

    assert outcomes == [1, 0, 1]


def test_mean_average_precision():
    """Verify mAP is the arithmetic mean of per-class AP scores."""
    ap_scores = [0.81, 0.74, 0.90]

    map_score = mean_average_precision(ap_scores)

    assert round(map_score, 4) == 0.8167


def test_calculate_map():
    """Verify per-class AP and overall mAP across multiple classes."""
    predictions = [
        {
            "label": "flour",
            "confidence": 0.95,
            "box": (100, 100, 300, 300),
        },
        {
            "label": "sugar",
            "confidence": 0.90,
            "box": (400, 100, 600, 300),
        },
        {
            "label": "milk",
            "confidence": 0.85,
            "box": (700, 100, 900, 300),
        },
        {
            "label": "milk",
            "confidence": 0.70,
            "box": (950, 100, 1100, 300),
        },
    ]

    ground_truths = [
        {
            "label": "flour",
            "box": (110, 110, 290, 290),
        },
        {
            "label": "sugar",
            "box": (410, 110, 590, 290),
        },
        {
            "label": "milk",
            "box": (710, 110, 890, 290),
        },
    ]

    map_score, ap_by_class = calculate_map(
        predictions,
        ground_truths,
        iou_threshold=0.5,
    )

    assert round(ap_by_class["flour"], 4) == 1.0
    assert round(ap_by_class["sugar"], 4) == 1.0
    assert round(ap_by_class["milk"], 4) == 1.0
    assert round(map_score, 4) == 1.0

def test_xyxy_to_yolo():
    """
    Verify XYXY boxes are converted to normalized YOLO coordinates.
    """
    box = (200, 100, 600, 300)

    result = xyxy_to_yolo(
        box,
        image_width=1000,
        image_height=500,
    )

    assert result == (0.4, 0.4, 0.4, 0.4)