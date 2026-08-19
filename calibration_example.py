import numpy as np


def apply_group_calibration(
    pred_cal_train,
    y_cal_train,
    group_cal_train,
    pred_cal_holdout,
    y_cal_holdout,
    group_cal_holdout,
    pred_target,
    group_target,
    min_cal_train_points=35,
    min_cal_holdout_points=35,
    min_relative_kl_improvement=0.02,
    shrinkage_strength=80.0,
    calibrator_clip_min=0.60,
    calibrator_clip_max=1.60,
    max_accepted_calibrators=30,
):
    """Apply subgroup-level multiplicative calibration to class probabilities.

    Parameters
    ----------
    pred_cal_train : array, shape (n_train, n_classes)
        Base model probabilities on calibration-train rows.
    y_cal_train : array, shape (n_train,)
        True class labels for calibration-train rows, encoded as 0..K-1.
    group_cal_train : array, shape (n_train,)
        Group id per calibration-train row. Can be tuple, int, str, etc.
    pred_cal_holdout : array, shape (n_holdout, n_classes)
        Base model probabilities on calibration-holdout rows.
    y_cal_holdout : array, shape (n_holdout,)
        True class labels for calibration-holdout rows, encoded as 0..K-1.
    group_cal_holdout : array, shape (n_holdout,)
        Group id per calibration-holdout row.
    pred_target : array, shape (n_target, n_classes)
        Base model probabilities to calibrate.
    group_target : array, shape (n_target,)
        Group id per target row.

    Returns
    -------
    pred_target_calibrated : array, shape (n_target, n_classes)
        Calibrated probabilities after subgroup-wise multiplicative correction.
    map_calibrators : dict
        Mapping {group_id -> classwise multiplier vector} for accepted groups.
    """

    pred_cal_train = np.asarray(pred_cal_train, dtype=float)
    y_cal_train = np.asarray(y_cal_train, dtype=int)
    group_cal_train = np.asarray(group_cal_train, dtype=object)

    pred_cal_holdout = np.asarray(pred_cal_holdout, dtype=float)
    y_cal_holdout = np.asarray(y_cal_holdout, dtype=int)
    group_cal_holdout = np.asarray(group_cal_holdout, dtype=object)

    pred_target = np.asarray(pred_target, dtype=float)
    group_target = np.asarray(group_target, dtype=object)

    n_classes = pred_cal_train.shape[1]

    def distribution(y):
        counts = np.bincount(y, minlength=n_classes).astype(float)
        return counts / np.sum(counts)

    def kl_divergence(pred_dist, true_dist, eps=1e-9):
        p = np.asarray(pred_dist, dtype=float)
        q = np.asarray(true_dist, dtype=float)
        return np.sum(q * np.log((q + eps) / (p + eps)))

    def simple_calibrator(predictions, target_dist):
        n_rows, n_cols = predictions.shape
        x = np.zeros((n_cols,), dtype=float)
        for c in range(n_cols):
            denom = (1.0 / n_rows) * np.sum(predictions[:, c]) + 1e-9
            x[c] = target_dist[c] / denom
        return x

    def normalize_rows(pred):
        pred = np.clip(pred, 0.0, None)
        row_sums = np.sum(pred, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return pred / row_sums

    candidate_calibrators = []
    unique_groups = np.unique(group_cal_train)

    for group_id in unique_groups:
        idx_train = group_cal_train == group_id
        n_train = int(np.sum(idx_train))
        if n_train < min_cal_train_points:
            continue

        pred_train_group = pred_cal_train[idx_train]
        y_train_group = y_cal_train[idx_train]

        x_raw = simple_calibrator(pred_train_group, distribution(y_train_group))

        # Shrink and clip to avoid unstable corrections in smaller groups.
        shrink = n_train / (n_train + shrinkage_strength)
        x = 1.0 + shrink * (x_raw - 1.0)
        x = np.clip(x, calibrator_clip_min, calibrator_clip_max)

        idx_holdout = group_cal_holdout == group_id
        n_holdout = int(np.sum(idx_holdout))
        if n_holdout < min_cal_holdout_points:
            continue

        pred_holdout_group = pred_cal_holdout[idx_holdout]
        y_holdout_group = y_cal_holdout[idx_holdout]

        pred_holdout_cal = normalize_rows(pred_holdout_group * x)

        true_dist = distribution(y_holdout_group)
        kl_before = kl_divergence(np.mean(pred_holdout_group, axis=0), true_dist)
        kl_after = kl_divergence(np.mean(pred_holdout_cal, axis=0), true_dist)
        rel_gain = (kl_before - kl_after) / (kl_before + 1e-9)

        if kl_after < kl_before and rel_gain >= min_relative_kl_improvement:
            candidate_calibrators.append((group_id, x, kl_before - kl_after))

    candidate_calibrators.sort(key=lambda t: -t[2])
    selected = candidate_calibrators[:max_accepted_calibrators]
    map_calibrators = {group_id: x for group_id, x, _ in selected}

    pred_target_calibrated = pred_target.copy()
    for group_id, x in map_calibrators.items():
        idx_target = group_target == group_id
        pred_target_calibrated[idx_target] *= x

    pred_target_calibrated = normalize_rows(pred_target_calibrated)
    return pred_target_calibrated, map_calibrators
