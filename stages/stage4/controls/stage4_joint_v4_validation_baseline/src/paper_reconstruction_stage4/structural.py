"""Observable-only structural evidence. Heuristic terms are not confidence.

Images/masks/contours and supplied candidate transforms are the only inputs.
Text axes are undirected and confidence-gated; no OCR, reference page or truth.
"""
from collections import OrderedDict, defaultdict
import time
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, gaussian_filter1d, map_coordinates
from scipy.spatial import ConvexHull, QhullError, cKDTree
from paper_reconstruction_contracts import map_points
from .geometry import probes, relative


def image_profile(piece):
    with Image.open(piece.image_path) as image:
        image = image.convert("L")
        scale = min(1., 320/max(image.size))
        image = image.resize((max(1, round(image.width*scale)), max(1, round(image.height*scale))))
        gray = np.asarray(image, np.float32)/255.
    mask = np.asarray(Image.fromarray(piece.mask).resize(image.size, Image.Resampling.NEAREST), bool)
    interior = binary_erosion(mask, iterations=2)
    if interior.sum() < 64:
        interior = mask
    values = gray[interior]
    paper = float(np.quantile(values, .85)) if len(values) else 1.
    dark = float(np.quantile(values, .02)) if len(values) else paper
    ink = np.clip((paper-gray)/max(paper-dark, .1), 0., 1.)
    ink[~mask] = 0.
    profile = {"angle_radians": 0., "axis_confidence": 0., "line_pitch_native_px": None, "pitch_confidence": 0.,
               "ink": ink.astype(float), "scale_xy": np.array([image.width/piece.shape[1], image.height/piece.shape[0]])}
    if len(values) < 64 or paper-dark < .04 or float(ink[interior].std()) < .025:
        profile["reason"] = "insufficient_interior_contrast"
        return profile
    y, x = np.nonzero(interior)
    take = np.linspace(0, len(x)-1, min(30000, len(x))).astype(int)
    x, y = x[take], y[take]
    weights = ink[y, x]
    scores, signals = [], []
    for angle in np.deg2rad(np.arange(0., 180., 2.)):
        coordinate = -np.sin(angle)*x+np.cos(angle)*y
        bins = np.rint(coordinate-coordinate.min()).astype(int)
        count = np.bincount(bins)
        signal = np.bincount(bins, weights=weights)/np.maximum(count, 1)
        valid = count >= max(6, .15*count.max())
        signal[~valid] = 0.
        signal = signal-gaussian_filter1d(signal, 3.)
        scores.append(float(np.mean(signal[valid]**2)) if valid.any() else 0.)
        signals.append(signal)
    best = int(np.argmax(scores))
    ratio = scores[best]/max(float(np.median(scores)), 1e-8)
    confidence = float(np.clip((ratio-4.)/12., 0., 1.))
    profile.update(angle_radians=float(np.deg2rad(best*2)), axis_confidence=confidence,
                   axis_peak_to_median_ratio=ratio, reason="observed_projection_axis" if confidence else "ambiguous_projection_axis")
    signal = signals[best]
    correlation = np.correlate(signal, signal, mode="full")[len(signal)-1:]
    if confidence and correlation[0] > 1e-8:
        correlation /= correlation[0]
        limit = min(64, len(signal)//4)
        peaks = [i for i in range(3, limit-1) if correlation[i] >= correlation[i-1] and correlation[i] > correlation[i+1]]
        if peaks:
            maximum = max(correlation[i] for i in peaks)
            if maximum > .25:
                lag = min(i for i in peaks if correlation[i] >= max(.25, .85*maximum))
                profile.update(line_pitch_native_px=float(lag/np.mean(profile["scale_xy"])),
                               pitch_confidence=float(confidence*np.clip(maximum, 0., 1.)))
    return profile


def jacobian_axes(h, shape, angle):
    center = np.array([shape[1]/2, shape[0]/2])
    direction = np.array([np.cos(angle), np.sin(angle)])
    normal = np.array([-direction[1], direction[0]])
    transformed = map_points([center, center+direction, center+normal], h)
    return transformed[1]-transformed[0], transformed[2]-transformed[0]


def sample_ink(profile, points):
    xy = np.asarray(points)*profile["scale_xy"]
    return map_coordinates(profile["ink"], [xy[:, 1], xy[:, 0]], order=1, mode="constant", cval=0.)


class StructuralScorer:
    def __init__(self, problem, config, support):
        started = time.perf_counter()
        self.problem, self.config, self.support = problem, config, support
        self.edges = {e.id: e for e in problem.edges}
        self.profiles = {p: image_profile(piece) for p, piece in problem.pieces.items()}
        self.contours = {}
        self.normals = {}
        for p, piece in problem.pieces.items():
            xy = piece.contour["points_xy"]
            self.contours[p] = xy[np.linspace(0, len(xy)-1, min(48, len(xy))).astype(int)]
            self.normals[p] = cKDTree(xy)
        self.pair_maps = defaultdict(list)
        for edge in problem.edges:
            points = probes(problem.pieces[edge.b].shape, 3)
            try:
                self.pair_maps[edge.a, edge.b].append((edge.id, map_points(points, edge.transform), float(np.clip(edge.weight, 0., 1.))))
            except ValueError:
                continue
        self.pair_maps = {pair: (ids, np.asarray(xy), np.asarray(weights)) for pair, values in self.pair_maps.items() if values
                          for ids, xy, weights in [zip(*values)]}
        self.cache = OrderedDict()
        self.cache_hits = self.evaluations = 0
        self.setup_seconds = time.perf_counter()-started

    def profile_report(self):
        return {p: {k: v for k, v in profile.items() if k not in {"ink", "scale_xy"}} for p, profile in self.profiles.items()}

    def boundary_residual(self, state, sample_count=17, appearance=True):
        boundary, visual = [], []
        for eid in state.selected:
            oid = state.assignments.get(eid)
            if oid not in self.support.arcs:
                continue
            arcs = self.support.arcs_for(state, eid) if hasattr(self.support, "arcs_for") else self.support.arcs[oid]
            for first, second in (arcs, arcs[::-1]):
                points = first.points[np.linspace(0, len(first.points)-1, sample_count).astype(int)]
                target = second.points
                h = relative(state.poses, second.piece, first.piece)
                mapped = map_points(points, h)
                _, nearest = cKDTree(target).query(mapped)
                # Native residuals are bidirectional, with no reference anchors.
                boundary.extend(((mapped-target[nearest])/self.config.contact_tolerance_px).ravel())
                if appearance and self.config.structural_appearance_weight:
                    piece = self.problem.pieces[first.piece]
                    _, indices = self.normals[first.piece].query(points)
                    normals = piece.contour["inward_normal_xy"][indices]
                    a = points+normals*3.
                    b = map_points(points-normals*3., h)
                    profile_a, profile_b = self.profiles[first.piece], self.profiles[second.piece]
                    values_a, values_b = sample_ink(profile_a, a), sample_ink(profile_b, b)
                    # Keep out-of-mask samples as a bounded penalty, not a zero
                    # residual that could reward opening a gap during fitting.
                    xy = np.rint(b).astype(int)
                    mask = self.problem.pieces[second.piece].mask
                    valid = (xy[:, 0] >= 0) & (xy[:, 1] >= 0) & (xy[:, 0] < mask.shape[1]) & (xy[:, 1] < mask.shape[0])
                    inside = np.zeros(len(xy), bool)
                    inside[valid] = mask[xy[valid, 1], xy[valid, 0]]
                    visual.extend(np.where(inside, values_a-values_b, .5))
        return np.asarray(boundary), np.asarray(visual)

    def evaluate(self, state):
        key = (tuple((p, state.groups[p], state.poses[p].tobytes()) for p in sorted(state.poses)), state.selected,
               tuple(sorted(state.assignments.items())))
        if state.active_support:
            key = (*key, tuple(sorted((e, a["support_id"]) for e, a in state.active_support.items())))
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        self.evaluations += 1
        config = self.config
        selected = {tuple(sorted((self.edges[e].a, self.edges[e].b))) for e in state.selected}
        gain = sum(float(np.clip(self.edges[e].weight, -1., 1.))-config.join_cost for e in state.selected)
        uncertainty = sum(bool(self.edges[e].raw.get("pose", {}).get("alignment", {}).get("underconstrained", False)) for e in state.selected)
        angle_cost = scale_cost = compactness_cost = corroboration = 0.
        corroborating = []
        for component in state.components():
            if len(component) < 2:
                continue
            angles, pitches = [], []
            for p in component:
                profile = self.profiles[p]
                u, v = jacobian_axes(state.poses[p], self.problem.pieces[p].shape, profile["angle_radians"])
                if profile["axis_confidence"]:
                    angles.append((float(np.arctan2(u[1], u[0])), profile["axis_confidence"]))
                if profile["line_pitch_native_px"] and profile["pitch_confidence"]:
                    pitches.append((np.log(max(np.linalg.norm(v)*profile["line_pitch_native_px"], 1e-8)), profile["pitch_confidence"]))
            if len(angles) >= 2:
                mean = .5*np.angle(sum(w*np.exp(2j*a) for a, w in angles))
                sigma = np.deg2rad(config.structural_angle_sigma_degrees)
                component_angle = sum(w*min((.5*np.angle(np.exp(2j*(a-mean)))/sigma)**2, 4.) for a, w in angles)
                # These are weak component plausibility priors. Correlated page
                # styles must not multiply an uncertain assumption by fragment
                # count. Preserve the historical sum in v2/v3 ablations.
                angle_cost += component_angle/len(angles) if config.objective_version == "observable-joint.v4" else component_angle
            if len(pitches) >= 2:
                # Text projection can select alternate lines or double line
                # spacing. Fold octave aliases instead of confidently treating
                # a 2x pitch estimate as a 2x geometric scale error.
                octave = np.log(2.)
                phase = np.angle(sum(w*np.exp(2j*np.pi*a/octave) for a, w in pitches))
                component_scale = sum(w*min((np.angle(np.exp(1j*(2*np.pi*a/octave-phase)))*octave/(2*np.pi*config.structural_scale_sigma))**2, 4.) for a, w in pitches)
                scale_cost += component_scale/len(pitches) if config.objective_version == "observable-joint.v4" else component_scale
            if config.structural_compactness_weight:
                clouds, area = [], 0.
                for p in component:
                    h = state.poses[p]
                    clouds.append(map_points(self.contours[p], h))
                    u, v = jacobian_axes(h, self.problem.pieces[p].shape, 0.)
                    area += abs(float(np.linalg.det(np.column_stack((u, v)))))*self.problem.pieces[p].mask.sum()
                try:
                    hull_area = ConvexHull(np.concatenate(clouds)).volume
                    compactness_cost += (len(component)-1)*max(0., 1.-area/max(hull_area, 1.))
                except QhullError:
                    pass
        if config.structural_corroboration_weight:
            for (a, b), (ids, xy, weights) in self.pair_maps.items():
                if a not in state.groups or b not in state.groups:
                    continue
                if state.groups[a] != state.groups[b] or (a, b) in selected:
                    continue
                points = probes(self.problem.pieces[b].shape, 3)
                actual = map_points(points, relative(state.poses, a, b))
                errors = np.sqrt(np.mean(np.sum((xy-actual)**2, axis=2), axis=1))
                sigma = max(4., config.structural_corroboration_fraction*np.linalg.norm(self.problem.pieces[a].shape))
                values = weights*np.exp(-.5*(errors/sigma)**2)
                best = int(np.argmax(values))
                corroboration += float(values[best])
                if values[best] > .25:
                    corroborating.append({"pieces": [a, b], "geometry_id": ids[best], "crop_probe_rms_px": float(errors[best])})
        boundary, appearance = self.boundary_residual(state)
        boundary_cost = float(np.mean(np.minimum(boundary**2, 4.)))*len(state.selected) if len(boundary) else 0.
        appearance_cost = float(np.mean(appearance**2))*len(state.selected) if len(appearance) else 0.
        penalties = {"text_axis": config.structural_angle_weight*angle_cost, "text_pitch": config.structural_scale_weight*scale_cost,
                     "compactness": config.structural_compactness_weight*compactness_cost,
                     "boundary": config.structural_boundary_weight*boundary_cost, "appearance": config.structural_appearance_weight*appearance_cost,
                     "underconstrained": config.structural_underconstrained_cost*uncertainty}
        report = {"score": float(gain+config.structural_corroboration_weight*corroboration-sum(penalties.values())),
                  "selected_edge_gain": gain, "penalties": penalties, "corroboration_reward": config.structural_corroboration_weight*corroboration,
                  "corroborating_unselected_pairs": corroborating, "confidence_calibrated": False}
        if len(self.cache) >= config.maximum_cache_entries:
            self.cache.popitem(last=False)
        self.cache[key] = report
        return report
