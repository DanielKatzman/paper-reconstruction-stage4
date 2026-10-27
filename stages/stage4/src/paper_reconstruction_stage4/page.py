"""Observed-only complete-page retention and assessment, frozen policy v1.

No page format, DPI, truth coordinates, RGB background or evaluation reader is
used here. Raster measurements are bounded approximations, not certificates.
"""
from collections import Counter, OrderedDict
import time
import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes
from scipy.spatial import ConvexHull
from paper_reconstruction_contracts import map_points
from .constraints import POLICY, rigid_check
from .geometry import relative
from .search.execution import clone, state_key
from .support import overlaps, contact_error

LIMITS = {'retained_page_states': 32, 'completed_feature_checks': 4096,
          'final_native_checks': 64, 'cache_entries': 512,
          'coarse_raster_side': 256, 'fine_raster_side': 768,
          'alternative_contact_checks': 256}
THRESHOLDS = {'minimum_fill': .88, 'maximum_holes': .035, 'maximum_overlap': .08,
              'alternative_score_margin': .025, 'alternative_displacement_fraction': .05}


def make_page_context(problem, config, support):
    from .config import REPAIR_POLICY, REPAIR_V15_POLICIES, REPAIR_V16_POLICY, REPAIR_V17_POLICY, REPAIR_V18_POLICY
    if config.native_repair_policy == REPAIR_V18_POLICY:
        if config.native_cut_repair:
            if (problem.capture_metadata or {}).get('rotation_family')=='continuous_rigid':
                from .native_capture_repair import RepairContext
                return RepairContext(problem,config,support)
            from .native_repair_v21 import RepairContext
            return RepairContext(problem,config,support)
        from .native_repair_v18 import RepairContext
        return RepairContext(problem,config,support)
    if config.native_repair_policy == REPAIR_V17_POLICY:
        from .native_repair_v17 import RepairContext
        return RepairContext(problem,config,support)
    if config.native_repair_policy == REPAIR_V16_POLICY:
        from .native_repair_v16 import RepairContext
        return RepairContext(problem,config,support)
    if config.native_repair_policy in REPAIR_V15_POLICIES:
        from .native_repair_v15 import RepairContext
        return RepairContext(problem,config,support)
    if config.native_repair_policy == REPAIR_POLICY:
        from .native_repair import RepairContext
        return RepairContext(problem,config,support)
    from .config import GEOMETRY_SELECTION, AMBIGUITY_SELECTION, CLOSURE_POLICY
    if config.component_closure_policy == CLOSURE_POLICY:
        from .component_closure import ClosureContext
        return ClosureContext(problem, config, support)
    if config.page_selection_policy == AMBIGUITY_SELECTION:
        from .ambiguity import AmbiguityContext
        return AmbiguityContext(problem, config, support)
    if config.page_selection_policy == GEOMETRY_SELECTION:
        from .page_geometry import GeometryContext
        return GeometryContext(problem, config, support)
    return PageContext(problem, config, support)


def enclosing_rectangle(points):
    """Minimum-area rectangle of pixel centres; pad by a native pixel cell."""
    hull = points[ConvexHull(points).vertices]
    edges = np.roll(hull, -1, axis=0) - hull
    angles = np.unique(np.mod(np.arctan2(edges[:, 1], edges[:, 0]), np.pi/2))
    best = None
    for angle in angles:
        c, s = np.cos(angle), np.sin(angle)
        axes = np.array([[c, -s], [s, c]])
        xy = hull @ axes
        lo, hi = xy.min(axis=0)-.5, xy.max(axis=0)+.5
        area = float(np.prod(hi-lo))
        if best is None or area < best[0]-1e-6:
            best = (area, axes, lo, hi)
    return best


def union_intervals(intervals):
    result = []
    for lo, hi in sorted(intervals):
        if result and lo <= result[-1][1]+1e-7:
            result[-1][1] = max(hi, result[-1][1])
        else:
            result.append([float(lo), float(hi)])
    return result


def uncovered_intervals(circumference, intervals):
    missing, start = [], 0.
    for lo, hi in union_intervals(intervals):
        if lo > start:
            missing.append([start, lo])
        start = max(start, hi)
    if start < circumference:
        missing.append([start, circumference])
    return missing


class PageContext:
    def __init__(self, problem, config, support):
        self.problem, self.config, self.support = problem, config, support
        self.hulls, self.areas, self.perimeters, self.images = {}, {}, {}, {}
        self.cache = OrderedDict(); self.counts = Counter(); self.seconds = 0.
        self.edges = {e.id: e for e in problem.edges}
        for pid, piece in problem.pieces.items():
            xy = piece.contour['points_xy']
            self.hulls[pid] = xy[ConvexHull(xy).vertices]
            self.areas[pid] = int(piece.mask.sum())
            self.perimeters[pid] = float(np.linalg.norm(np.roll(xy, -1, axis=0)-xy, axis=1).sum())
            self.images[pid] = Image.fromarray(piece.mask.astype(np.uint8)*255)

    def quick(self, state):
        complete = len(state.components()) == 1
        invalid = [pid for pid, h in state.poses.items() if not rigid_check(h)['valid']]
        if invalid or not complete:
            return {'complete': complete, 'rigid': not invalid, 'invalid_pieces': invalid,
                    'gap_proxy': 0., 'overlap_proxy': 0.}
        points = np.vstack([map_points(self.hulls[p], state.poses[p]) for p in sorted(state.poses)])
        area, _, _, _ = enclosing_rectangle(points)
        total = sum(self.areas.values())
        return {'complete': True, 'rigid': True, 'rectangle_area_px2': area,
                'native_area_px2': total, 'gap_proxy': max(0., 1-total/area),
                'overlap_proxy': max(0., 1-area/total)}

    def utility(self, state, features):
        n = max(1, len(state.poses)-1)
        if not features['rigid']:
            return -1e12
        if not features['complete']:
            return state.score  # Missing regions of partial forests are not holes.
        gap = features.get('unfilled_fraction', features['gap_proxy'])
        overlap = features.get('overlap_fraction', features['overlap_proxy'])
        weak = features.get('contacts', {}).get('unresolved_weak_bridge_fraction', 0.)
        return float(state.score - n*(2.*gap+overlap+.15*weak))

    def closure_priority(self, state, features):
        """A plausible complete page takes priority over a partial grouping.

        The secondary utility remains comparable within each class. This is a
        declared selection rule, not an artificial gap penalty on partials.
        Alternative disagreement is still assessed after selecting candidates.
        """
        if self.config.page_selection_policy != 'supported-closure-first.v1':
            return 0
        if not features['complete'] or not features['rigid']:
            return 0
        return int(features.get('fill_fraction',1-features['gap_proxy']) >= THRESHOLDS['minimum_fill'] and
            features.get('overlap_fraction',features['overlap_proxy']) <= THRESHOLDS['maximum_overlap'] and
            features.get('interior_hole_fraction',0.) <= THRESHOLDS['maximum_holes'] and
            not features.get('contacts',{}).get('unresolved_weak_geometry_ids') and
            len(state.selected) >= len(state.poses)-1)

    def raster(self, state, side):
        points = np.vstack([map_points(self.hulls[p], state.poses[p]) for p in sorted(state.poses)])
        area, axes, lo, hi = enclosing_rectangle(points)
        scale = min(1., (side-4)/float(max(hi-lo)))
        size = np.maximum(1, np.ceil((hi-lo)*scale).astype(int))
        canvas_shape = (int(size[1]+4), int(size[0]+4))
        world_to_raster = np.eye(3)
        world_to_raster[:2, :2] = scale*axes.T
        world_to_raster[:2, 2] = 2-scale*lo
        counts = np.zeros(canvas_shape, np.uint16)
        for pid in sorted(state.poses):
            inverse = np.linalg.inv(world_to_raster @ state.poses[pid])
            warped = self.images[pid].transform((canvas_shape[1], canvas_shape[0]), Image.Transform.AFFINE,
                        tuple(inverse[:2].ravel()), resample=Image.Resampling.NEAREST)
            counts += np.asarray(warped) > 127
        foreground = counts > 0
        union = int(foreground.sum()); total = int(counts.sum())
        raster_rectangle_area = area*scale*scale
        return {'rectangle_area_px2': area, 'rectangle_sides_px': sorted((hi-lo).tolist()),
                'union_area_px2_estimate': union/(scale*scale),
                'fill_fraction': min(1., union/max(1., raster_rectangle_area)),
                'unfilled_fraction': max(0., 1-union/max(1., raster_rectangle_area)),
                'overlap_fraction': max(0., (total-union)/max(1, total)),
                'interior_hole_fraction': float((binary_fill_holes(foreground) & ~foreground).sum()/max(1., raster_rectangle_area)),
                'raster_side_limit': side, 'raster_shape_hw': list(canvas_shape),
                'native_pixels_per_raster_pixel': 1/scale,
                'raster_fraction_uncertainty_estimate': min(1., 2*float(size.sum())/max(1., raster_rectangle_area)),
                'orientation_freely_fitted': True, 'aspect_ratio_prior_used': False}

    def contacts(self, state):
        occupied = {p: [] for p in state.poses}; used = []; joins = []
        selected = set(state.selected)
        for eid in state.selected:
            edge = self.edges[eid]
            arcs = (self.support.arcs_for(state, eid) if hasattr(self.support, 'arcs_for')
                    else self.support.arcs[state.assignments[eid]])
            used.extend(arcs)
            for arc in arcs:
                occupied[arc.piece].extend(arc.intervals)
            length = min(a.length for a in arcs)
            spreads = []
            for arc in arcs:
                singular = np.linalg.svd(arc.points-arc.points.mean(axis=0), compute_uv=False)
                spreads.append(float(singular[-1]/max(singular[0], 1e-9)))
            spread = min(spreads)
            # Direction agreement does not constrain slide along a straight seam.
            short = bool(length < .12*np.sqrt(min(self.areas[edge.a], self.areas[edge.b])))
            weak = bool(short and spread < .05)
            # Find the selected-graph cut, without assuming a rectangular grid.
            group = {edge.a}; changed = True
            while changed:
                changed = False
                for other in state.selected:
                    if other == eid:
                        continue
                    e = self.edges[other]
                    if (e.a in group) != (e.b in group):
                        group.update((e.a, e.b)); changed = True
            bridge = edge.b not in group
            joins.append({'geometry_id': eid, 'occurrence_id': state.assignments[eid],
                'pieces': [edge.a, edge.b], 'native_length_px': length,
                'native_length_over_sqrt_smaller_area': length/np.sqrt(min(self.areas[edge.a], self.areas[edge.b])),
                'shape_spread_ratio': spread, 'short_straight': weak, 'selected_graph_bridge': bridge,
                'native_spans': [{'piece_id': a.piece, 'intervals': [list(v) for v in a.intervals]} for a in arcs],
                'cut': sorted(group) if bridge else [], 'independent_witness_pairs': []})
        # Unselected hypotheses are witnesses only when their actual native spans
        # are consistent, unoccupied and mutually independent. One vote per pair.
        witnesses, checked = [], 0
        alternatives = sorted((e for e in self.problem.edges if e.id not in selected), key=lambda e: (-e.weight, e.id))
        for edge in alternatives:
            if checked >= LIMITS['alternative_contact_checks']:
                break
            if state.groups[edge.a] != state.groups[edge.b]:
                continue
            if any(set(w['pieces']) == {edge.a, edge.b} for w in witnesses):
                continue
            viable = []
            for o in edge.occurrences:
                arcs = self.support.arcs.get(o['occurrence_id'])
                if arcs is not None and not any(overlaps(a, b, self.config) for a in arcs for b in used):
                    viable.append((min(a.length for a in arcs), o['occurrence_id'], arcs))
            for _, oid, arcs in sorted(viable, key=lambda v: (-v[0], v[1]))[:1]:
                checked += 1
                if contact_error(arcs, state.poses) > self.config.contact_tolerance_px:
                    continue
                length = min(a.length for a in arcs)
                if length < .12*np.sqrt(min(self.areas[edge.a], self.areas[edge.b])):
                    continue
                witnesses.append({'pieces': [edge.a, edge.b], 'geometry_id': edge.id, 'occurrence_id': oid,
                                  'native_length_px': length})
                used.extend(arcs)
                for join in joins:
                    cut = set(join['cut'])
                    if cut and ((edge.a in cut) != (edge.b in cut)):
                        join['independent_witness_pairs'].append([edge.a, edge.b])
        weak_bridges = [j for j in joins if j['short_straight'] and j['selected_graph_bridge'] and not j['independent_witness_pairs']]
        return {'joins': joins, 'independent_witnesses': witnesses, 'alternative_checks': checked,
                'alternative_checks_truncated': checked >= LIMITS['alternative_contact_checks'],
                'unresolved_weak_bridge_fraction': len(weak_bridges)/max(1, len(joins)),
                'unresolved_weak_geometry_ids': [j['geometry_id'] for j in weak_bridges],
                'occupied_boundary_length_px': {p: sum(b-a for a, b in union_intervals(v)) for p, v in occupied.items()},
                'uncovered_boundary_intervals': {p: uncovered_intervals(self.perimeters[p], v) for p, v in occupied.items()},
                'uncovered_regions_include_outer_page_border': True,
                'text_axis_alignment_resolves_slide': False, 'duplicate_hypotheses_are_independent_votes': False}

    def features(self, state, fine=True):
        key = (state_key(state), fine)
        if key in self.cache:
            self.counts['cache_hits'] += 1; self.cache.move_to_end(key)
            return self.cache[key]
        started = time.perf_counter(); self.counts['feature_extractions'] += 1
        value = self.quick(state)
        if value['complete'] and value['rigid']:
            value.update(self.raster(state, LIMITS['fine_raster_side'] if fine else LIMITS['coarse_raster_side']))
            value['contacts'] = self.contacts(state)
            clouds = {p:map_points(self.hulls[p],state.poses[p]) for p in sorted(state.poses)}
            _, axes, lo, hi = enclosing_rectangle(np.vstack(list(clouds.values())))
            exposed = {}; extensions = {}
            tolerance = .005*np.sqrt(sum(self.areas.values()))
            for pid in clouds:
                piece = self.problem.pieces[pid]
                xy = piece.contour['points_xy']; circumference = self.perimeters[pid]
                cumulative = np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
                occupied = []
                for join in value['contacts']['joins']:
                    for span in join['native_spans']:
                        if span['piece_id'] == pid:
                            occupied.extend(span['intervals'])
                unoccupied = np.ones(len(xy),bool)
                for a,b in occupied:
                    unoccupied &= ~((cumulative >= a)&(cumulative <= b))
                positions = map_points(xy,state.poses[pid])@axes
                border_distance = np.minimum(positions-lo,hi-positions).min(axis=1)
                segment_lengths = np.linalg.norm(np.roll(xy,-1,axis=0)-xy,axis=1)
                interior = unoccupied & (border_distance > tolerance)
                exposed[pid] = float(segment_lengths[interior].sum()/max(1.,circumference))
                if len(clouds) > 1:
                    without = enclosing_rectangle(np.vstack([v for p,v in clouds.items() if p != pid]))[0]
                    extensions[pid] = float(max(0.,value['rectangle_area_px2']-without)/value['rectangle_area_px2'])
            value['unoccupied_interior_boundary_fraction_by_piece'] = exposed
            value['rectangle_extension_fraction_by_piece'] = extensions
            value['exposed_boundary_tolerance_native_px'] = tolerance
            value['exposed_boundary_is_diagnostic_not_a_hard_torn_edge_label'] = True
        if len(self.cache) >= LIMITS['cache_entries']:
            self.cache.popitem(last=False)
        self.cache[key] = value
        self.seconds += time.perf_counter()-started
        return value

    def assess(self, state, alternatives=()):
        f = self.features(state); reasons = []
        if not f['rigid']:
            status = 'profile_violation'; reasons.append('profile_nonrigid_transform')
        else:
            if not f['complete']:
                reasons.append('incomplete_candidate')
            else:
                if f['unfilled_fraction'] > 1-THRESHOLDS['minimum_fill']:
                    reasons.extend(['large_unfilled_page_area', 'outlying_fragment_or_misaligned_group'])
                if f['interior_hole_fraction'] > THRESHOLDS['maximum_holes']:
                    reasons.append('large_interior_hole')
                if f['overlap_fraction'] > THRESHOLDS['maximum_overlap']:
                    reasons.append('excessive_mask_overlap')
                if f['contacts']['unresolved_weak_geometry_ids']:
                    reasons.append('short_straight_ambiguous_attachment')
                if not state.selected and len(state.poses) > 1:
                    reasons.append('insufficient_native_support')
            status = 'unresolved' if reasons else 'supported'
        # Partial states still report occupied/uncovered native support.
        if not f['complete'] and f['rigid'] and 'contacts' not in f:
            f = {**f, 'contacts': self.contacts(state)}
        disagreements = []
        if f['complete'] and f['rigid']:
            utility = self.utility(state, f); ids = sorted(state.poses)
            for other in alternatives:
                of = self.features(other)
                if not of['complete'] or not of['rigid']:
                    continue
                delta = abs(utility-self.utility(other, of))/max(1, len(ids)-1)
                if delta > THRESHOLDS['alternative_score_margin']:
                    continue
                errors = []
                for i, a in enumerate(ids):
                    for b in ids[i+1:]:
                        xy = self.hulls[b]
                        errors.append(float(np.quantile(np.linalg.norm(map_points(xy, relative(state.poses, a, b))-
                            map_points(xy, relative(other.poses, a, b)), axis=1), .95)))
                displacement = max(errors, default=0.)/np.sqrt(sum(self.areas.values()))
                if displacement > THRESHOLDS['alternative_displacement_fraction']:
                    disagreements.append({'normalized_displacement': displacement, 'normalized_score_margin': delta})
            if disagreements and status != 'profile_violation':
                status = 'unresolved'; reasons.append('material_alternative_disagreement')
        return {'schema_version': 'stage4.reconstruction-assessment.v1', 'policy': POLICY,
                'status': status, 'reason_codes': sorted(set(reasons)), 'features': f,
                'alternative_disagreements': disagreements, 'confidence_calibrated': False,
                'unique_correctness_guaranteed': False, 'evaluation_truth_opened': False,
                'thresholds': THRESHOLDS, 'limits': LIMITS}

    def report(self):
        return {'policy': POLICY, 'counts': dict(self.counts), 'feature_seconds': self.seconds,
                'cached_entries': len(self.cache), 'limits': LIMITS, 'thresholds': THRESHOLDS,
                'partial_page_bounds_enabled': False, 'evaluation_truth_opened': False}


class PageBank:
    """Cheap page candidates enter before old score/check-limit pruning."""
    def __init__(self, context, deadline):
        self.context, self.deadline = context, deadline
        self.states = {}; self.seen = set(); self.counts = Counter(); self.origins = {}

    def consider(self, state, origin):
        if len(state.components()) != 1:
            return
        key = state_key(state)
        if key in self.seen:
            self.counts['duplicate_completions'] += 1; return
        if len(self.seen) >= LIMITS['completed_feature_checks'] or time.perf_counter() >= self.deadline:
            self.counts['feature_budget_abstentions'] += 1; return
        self.seen.add(key); self.counts['completed_states_measured'] += 1
        features = self.context.quick(state)
        candidate = clone(state)
        # Source routes keep their own objective; bank ranking always uses v8.
        candidate.score = self.context.problem.structural.evaluate(candidate)['score']
        rank = self.context.utility(candidate, features)
        self.states[key] = (rank, candidate)
        self.origins[key] = origin
        ranked = sorted(self.states, key=lambda k: (-self.states[k][0], k))
        # 24 page winners plus 8 distinct support/topology contenders.
        keep = ranked[:24]; signatures = {tuple(sorted(self.states[k][1].selected)) for k in keep}
        for k in ranked[24:]:
            signature = tuple(sorted(self.states[k][1].selected))
            if signature not in signatures and len(keep) < LIMITS['retained_page_states']:
                keep.append(k); signatures.add(signature)
        self.states = {k: self.states[k] for k in keep}
        self.origins = {k: self.origins[k] for k in keep}

    def finalists(self):
        return [v[1] for v in sorted(self.states.values(), key=lambda v: -v[0])]

    def report(self):
        return {'policy': POLICY, 'counts': dict(self.counts), 'retained': len(self.states),
                'retention_before_old_score_pruning': True, 'baseline_finalists_protected_separately': True,
                'origins': list(self.origins.values()), 'limits': LIMITS}


def prioritize_finalists(problem, config, states, feasibility):
    """Independent final physical/support checks precede page selection."""
    from .active_support import verify_result_support
    from .outputs import serialize_layout
    context = problem.page_context
    # Page contenders and baseline finalists are both present in this list.
    # Cheap ranking sets only check order; no unsafe layout can be published.
    quick = {state_key(s):context.quick(s) for s in states}
    ordered = sorted(states, key=lambda s: (-context.closure_priority(s,quick[state_key(s)]),
                    -context.utility(s,quick[state_key(s)]), state_key(s)))
    accepted, rejected = [], Counter(); checked = 0
    started = time.perf_counter()
    deadline = min(started+config.joint_finalize_seconds, problem.deadline or float('inf'))
    for source in ordered[:LIMITS['final_native_checks']]:
        if checked and time.perf_counter() >= deadline:
            break
        checked += 1
        check = feasibility.state(source, final=True)
        if not check['valid']:
            rejected[check['reason']] += 1; continue
        try:
            verify_result_support(problem, {'layouts': [serialize_layout(problem, source, 0, check, {'attempted': False})]}, config)
        except ValueError:
            rejected['conflicting_native_support'] += 1; continue
        features = context.features(source)
        accepted.append((context.closure_priority(source,features), context.utility(source, features), clone(source)))
    accepted.sort(key=lambda v: (-v[0], -v[1], state_key(v[2])))
    return [s for _, _, s in accepted], {'checked': checked, 'available': len(states),
        'rejections': dict(rejected), 'limit': LIMITS['final_native_checks'],
        'elapsed_seconds': time.perf_counter()-started,
        'cooperative_finalization_seconds': config.joint_finalize_seconds,
        'budget_abstentions': max(0,len(states)-checked),
        'unverified_states_published': False, 'page_candidates_rechecked': True}
