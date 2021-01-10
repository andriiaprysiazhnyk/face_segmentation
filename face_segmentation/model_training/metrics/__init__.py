from .iou import iou


def get_metric(metric_name):
    if metric_name == "iou":
        return iou
    else:
        raise ValueError(f"Metric [{metric_name}] not recognized.")
