

def box_area(x1, y1, x2, y2):
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    return width * height


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1: Bounding box in the format (x1, y1, x2, y2).
        box2: Bounding box in the format (x1, y1, x2, y2).

    Returns:
        IoU score as a float between 0.0 and 1.0.
    """
    intersect_x1 = max(box1[0], box2[0])
    intersect_y1 = max(box1[1], box2[1])
    intersect_x2 = min(box1[2], box2[2])
    intersect_y2 = min(box1[3], box2[3])

    intersect_area = box_area(
        intersect_x1,
        intersect_y1,
        intersect_x2,
        intersect_y2
    )

    union = box_area(*box1) + box_area(*box2) - intersect_area

    if union == 0:
        return 0.0

    return intersect_area / union


def filter_detections(detections, threshold):
    passing_detections = []

    for detection in detections:
        if detection["confidence"] >= threshold:
            passing_detections.append(detection)

    return passing_detections


def non_max_suppression(detections, iou_threshold):
    detections = sorted(
        detections,
        key=lambda d: d["confidence"],
        reverse=True
    )

    kept = []

    while detections:
        current = detections.pop(0)
        kept.append(current)

        remaining = []

        for detection in detections:
            if current["label"] != detection["label"]:
                remaining.append(detection)
                continue

            if calculate_iou(current["box"], detection["box"]) <= iou_threshold:
                remaining.append(detection)

        detections = remaining

    return kept

def is_match(prediction, ground_truth, iou_threshold=0.5):
    """
    Determine whether a prediction matches a ground-truth object.

    Args:
        prediction: Predicted object containing a label and bounding box.
        ground_truth: Ground-truth object containing a label and bounding box.
        iou_threshold: Minimum IoU required for a valid match.

    Returns:
        True if the labels match and IoU meets the threshold; otherwise False.
    """
    return (prediction["label"] == ground_truth["label"]
            and calculate_iou(
                prediction["box"], 
                ground_truth["box"])
                  >= iou_threshold
            )

def evaluate_detections(predictions, ground_truths, iou_threshold=0.5):
    """
    Count true positives, false positives, and false negatives.

    Args:
        predictions: List of predicted objects.
        ground_truths: List of ground-truth objects.
        iou_threshold: Minimum IoU required for a valid match.

    Returns:
        A tuple containing true positives, false positives, and false negatives.
    """
    true_positives = 0
    false_positives = 0

    matched_ground_truths = set()

    for prediction in predictions:
        matched = False

        for index, ground_truth in enumerate(ground_truths):
            if index in matched_ground_truths:
                continue

            if is_match(prediction, ground_truth, iou_threshold):
                true_positives += 1
                matched_ground_truths.add(index)
                matched = True
                break

        if not matched:
            false_positives += 1

    false_negatives = len(ground_truths) - len(matched_ground_truths)

    return true_positives, false_positives, false_negatives


def precision(tp, fp):
    """
    Calculate precision from true positives and false positives.

    Args:
        tp: Number of true positives.
        fp: Number of false positives.

    Returns:
        Precision as a float from 0.0 to 1.0.
    """
    denominator = tp + fp

    if denominator == 0:
        return 0.0

    return tp / denominator


def recall(tp, fn):
    """
    Calculate recall from true positives and false negatives.

    Args:
        tp: Number of true positives.
        fn: Number of false negatives.

    Returns:
        Recall as a float from 0.0 to 1.0.
    """
    denominator = tp + fn

    if denominator == 0:
        return 0.0

    return tp / denominator


def average_precision(outcomes, num_ground_truths):
    """
    Calculate a simplified Average Precision (AP) score.

    Args:
        outcomes: Ranked list where 1 represents a true positive and
            0 represents a false positive.
        num_ground_truths: Total number of real objects for the class.

    Returns:
        Average Precision as a float from 0.0 to 1.0.
    """
    true_positives = 0
    false_positives = 0

    precisions = []
    recalls = []

    for outcome in outcomes:
        if outcome == 1:
            true_positives += 1
        else:
            false_positives += 1

        current_precision = true_positives / (true_positives + false_positives)
        current_recall = true_positives / num_ground_truths

        precisions.append(current_precision)
        recalls.append(current_recall)

    ap = 0.0
    previous_recall = 0.0

    for precision_value, recall_value in zip(precisions, recalls):
        recall_change = recall_value - previous_recall

        if recall_change > 0:
            ap += precision_value * recall_change
            previous_recall = recall_value

    return ap


def build_tp_fp_outcomes(predictions, ground_truths, iou_threshold=0.5):
    """
    Build ranked true-positive / false-positive outcomes for AP calculation.

    Args:
        predictions: List of predicted objects containing label, confidence,
            and bounding box.
        ground_truths: List of ground-truth objects containing label and box.
        iou_threshold: Minimum IoU required for a valid match.

    Returns:
        A ranked list where 1 represents a true positive and 0 represents
        a false positive.
    """

    sorted_predictions = sorted(predictions, key=lambda prediction: prediction["confidence"], reverse=True)

    matched_ground_truths = set()
    outcomes = []

    for prediction in sorted_predictions:
        matched = False

        for index, ground_truth, in enumerate(ground_truths):
            if index in matched_ground_truths:
                continue

            if is_match(prediction, ground_truth, iou_threshold):
                matched_ground_truths.add(index)
                matched = True
                break

        if matched:
            outcomes.append(1)
        else:
            outcomes.append(0)
        
    return outcomes


def mean_average_precision(ap_scores):
    """
    Calculate mean Average Precision (mAP) from per-class AP scores.

    Args:
        ap_scores: Iterable of Average Precision scores.

    Returns:
        Mean Average Precision as a float from 0.0 to 1.0.
    """
    if not ap_scores:
        return 0.0

    return sum(ap_scores) / len(ap_scores)


def calculate_map(predictions, ground_truths, iou_threshold=0.5):
    """
    Calculate mean Average Precision across all ground-truth classes.

    Args:
        predictions: List of predicted objects containing label, confidence,
            and bounding box.
        ground_truths: List of ground-truth objects containing label and box.
        iou_threshold: Minimum IoU required for a valid match.

    Returns:
        A tuple containing:
            - mAP score.
            - Dictionary of AP scores by class.
    """
    if len(ground_truths) == 0:
        return 0.0

    classes = set()

    for ground_truth in ground_truths:
        classes.add(ground_truth["label"])

    ap_by_class = {}

    for class_name in classes:
        # 1. Get predictions for this class
        class_predictions = []
        class_ground_truths = []

        for prediction in predictions:
            if prediction["label"] == class_name:
                class_predictions.append(prediction)

        # 2. Get ground truths for this class
        for ground_truth in ground_truths:
            if ground_truth["label"] == class_name:
                class_ground_truths.append(ground_truth)
        
        # 3. Build ranked TP/FP outcomes
        outcomes = build_tp_fp_outcomes(class_predictions, class_ground_truths, iou_threshold)

        # 4. Calculate AP
        class_ap = average_precision(outcomes, len(class_ground_truths))

        # 5. Store it in ap_by_class
        ap_by_class[class_name] = class_ap

    map_score = mean_average_precision(
        list(ap_by_class.values())
    )

    return map_score, ap_by_class


def xyxy_to_yolo(box, image_width, image_height):
    """
    Convert bounding-box coordinates from XYXY format to normalized YOLO format.

    Args:
        box: Bounding box in the format (x1, y1, x2, y2), where
            (x1, y1) is the top-left corner and (x2, y2) is the
            bottom-right corner.
        image_width: Width of the image in pixels.
        image_height: Height of the image in pixels.

    Returns:
        A tuple containing normalized YOLO coordinates in the format
        (center_x, center_y, width, height), with values ranging
        from 0.0 to 1.0.
    """
    box_width = (box[2] - box[0]) / image_width
    box_height = (box[3] - box[1]) / image_height

    center_x = ((box[2] + box[0]) / 2) / image_width
    center_y = ((box[3] + box[1]) / 2) / image_height

    return center_x, center_y, box_width, box_height

