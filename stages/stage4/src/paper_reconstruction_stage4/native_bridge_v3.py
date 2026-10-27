"""V18 bridge evidence: existing checks plus a replayable page-edge cut witness."""
import numpy as np
from paper_reconstruction_contracts import map_points
from . import native_bridge_v2 as previous
from .outputs import digest, native_probes
from .geometry import relative
from .native_bridge import record_arcs, check_arcs
from .repair_support_v16 import reachable
from .page_boundary_v18 import verify

POLICY = 'stage4.native-component-bridge.v3'


def make_record(problem, state, contact, source_id, witness):
    value = previous.make_record(problem, state, contact, source_id)
    value.update(policy=POLICY, page_boundary_witness=witness)
    value['bridge_id'] = digest({k: v for k, v in value.items() if k != 'bridge_id'})
    return value


def validate_record(value):
    if value.get('policy') != POLICY or 'page_boundary_witness' not in value:
        raise ValueError('Invalid page-corroborated bridge policy')
    if value.get('bridge_id') != digest({k: v for k, v in value.items() if k != 'bridge_id'}):
        raise ValueError('Changed page-corroborated bridge identity')
    base = {k: v for k, v in value.items() if k != 'page_boundary_witness'}
    base['policy'] = previous.POLICY
    base['bridge_id'] = digest({k: v for k, v in base.items() if k != 'bridge_id'})
    previous.validate_record(base)


def verify_layout(problem, layout, config):
    from .config import REPAIR_V18_POLICY
    from .active_support import record_arc
    from .support import occurrence_arcs
    from .page import make_page_context
    from .component_closure import shape_measurement
    from .model import State
    from .feasibility import Feasibility
    joins = layout['selected_joins']; checked = 0
    poses = {p['piece_id']: np.asarray(p['native_to_component'], float) for p in layout['pieces']}
    edges = {e.id: e for e in problem.edges}
    for index, join in enumerate(joins):
        value = join.get('native_bridge', {})
        if value.get('policy') != POLICY:
            continue
        if config.native_repair_policy != REPAIR_V18_POLICY:
            raise ValueError('Page-boundary bridge requires the V18 development policy')
        validate_record(value)
        if problem.capture_metadata != value['capture_binding'] or value['capture_binding']['capture_profile_sha256'] != problem.constraint_profile['capture_profile_sha256']:
            raise ValueError('Native repair capture provenance differs from authenticated input')
        arcs = record_arcs(problem, value); check_arcs(problem, arcs, poses, config)
        a, b = value['pieces']; actual = relative(poses, a, b)
        probes = np.asarray(native_probes(problem.pieces[b]))
        if float(np.quantile(np.linalg.norm(map_points(probes, actual)-map_points(probes, value['proposed_transform_b_to_a']), axis=1), .95)) > config.cycle_tolerance_px:
            raise ValueError('Native repair proposal and composed whole-piece pose disagree')
        pairs = [(r['piece_a'], r['piece_b']) for r in joins]
        cut = reachable(poses, pairs, a, index)
        if b in cut:
            raise ValueError('Page-boundary witness must describe an actual selected graph cut')
        if reachable(poses, pairs, a) != set(problem.pieces):
            raise ValueError('Page-boundary witness requires one connected complete assembly')
        occupied = []
        for row in joins:
            if row.get('active_support'):
                occupied.extend(record_arc(v) for v in row['active_support']['arcs'])
            else:
                edge = edges[row['geometry_id']]
                occurrence = next(o for o in edge.occurrences if o['occurrence_id'] == row['occurrence_id'])
                occupied.extend(occurrence_arcs(problem, occurrence))
        verify(problem, poses, cut, occupied, config, value['page_boundary_witness'])
        state = State(poses, {p: min(poses) for p in poses})
        context = getattr(problem, 'page_context', None)
        if context is None or context.problem is not problem:
            from .active_support import ActiveSupportIndex
            context = make_page_context(problem, config, ActiveSupportIndex(problem, config))
        if not shape_measurement(context, state, 768)['shape_gate_pass'] or not Feasibility(problem, config).state(state, final=True)['valid']:
            raise ValueError('Page-boundary witness fails full page/physical validation')
        checked += 1
    return checked
