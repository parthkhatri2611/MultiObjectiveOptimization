"""Dependency-free answer and two-objective Pareto metrics."""
from collections import Counter
import math
import re
import string


def normalize(text):
    text = str(text).lower().translate(str.maketrans('', '', string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())


def answer_scores(prediction, reference):
    p, r = normalize(prediction), normalize(reference)
    em = float(p == r)
    if (p in {'yes', 'no', 'noanswer'} or r in {'yes', 'no', 'noanswer'}) and p != r:
        return {'exact_match': em, 'f1': 0.0}
    pt, rt = p.split(), r.split()
    if not pt or not rt:
        return {'exact_match': em, 'f1': em}
    overlap = sum((Counter(pt) & Counter(rt)).values())
    f1 = 2 * overlap / (len(pt) + len(rt))
    return {'exact_match': em, 'f1': f1}


def percentile(values, q):
    if not values:
        return None
    v = sorted(values)
    index = (len(v) - 1) * q
    lo, hi = math.floor(index), math.ceil(index)
    return v[lo] + (v[hi] - v[lo]) * (index - lo)


def pareto_indices(points):
    """Quality maximized, resource minimized. Retains equal-objective ties."""
    return [i for i, (q, c) in enumerate(points)
            if not any(q2 >= q and c2 <= c and (q2 > q or c2 < c)
                       for j, (q2, c2) in enumerate(points) if i != j)]


def hypervolume(points, resource_reference):
    """Normalized 2D HV, reference (quality=0, resource=resource_reference).

    Points outside the declared resource reference contribute zero; no per-run scaling.
    """
    if resource_reference <= 0:
        raise ValueError('resource_reference must be positive')
    valid = [(q, c) for q, c in points if 0 <= q <= 1 and 0 <= c < resource_reference]
    front = [valid[i] for i in pareto_indices(valid)]
    previous_quality, area = 0.0, 0.0
    for quality, cost in sorted(set(front)):
        area += (quality - previous_quality) * (resource_reference - cost)
        previous_quality = quality
    return area / resource_reference


def igd(points, reference, resource_scale):
    if not points or not reference:
        return None
    if resource_scale <= 0:
        raise ValueError('resource_scale must be positive')
    return sum(min(math.hypot(q - q2, (c - c2) / resource_scale)
                   for q2, c2 in points) for q, c in reference) / len(reference)
