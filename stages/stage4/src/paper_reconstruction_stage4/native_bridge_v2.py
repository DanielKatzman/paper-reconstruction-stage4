"""Multi-component seam provenance and independent conditioning/cycle checks."""
import numpy as np
from .outputs import digest
from .constraints import require_rigid, require_native_rotation
from .geometry import relative
from .native_bridge import arc_value,record_arcs,check_arcs
from .seam_poses import spread

POLICY='stage4.native-component-bridge.v2'


def make_record(problem,state,contact,source_id):
    a,b=contact['pieces'];arcs=sorted(contact['arcs'],key=lambda a:a.piece)
    value={'policy':POLICY,'input_manifest_sha256':problem.manifest['manifest_sha256'],
        'pieces':[a,b],'arcs':[arc_value(v) for v in arcs],
        'transform_b_to_a':relative(state.poses,a,b).tolist(),
        'source_candidate_id':source_id,'placement_proposal_id':contact['proposal_id'],
        'source_groups':contact['source_groups'],'merged_groups':contact['merged_groups'],
        'role':contact['role'],'proposed_transform_b_to_a':contact['proposed_transform_b_to_a'],
        'capture_binding':problem.capture_metadata,
        'producer':'stage4_observed_native_contours','upstream_pose_or_occurrence_claimed':False}
    return {**value,'bridge_id':digest(value)}


def validate_record(value):
    keys={'policy','input_manifest_sha256','pieces','arcs','transform_b_to_a','source_candidate_id',
        'placement_proposal_id','source_groups','merged_groups','role','proposed_transform_b_to_a',
        'capture_binding','producer','upstream_pose_or_occurrence_claimed','bridge_id'}
    if set(value)!=keys or value['policy']!=POLICY or value['producer']!='stage4_observed_native_contours' or value['upstream_pose_or_occurrence_claimed'] is not False:
        raise ValueError('Invalid multi-component native bridge provenance')
    if value['bridge_id']!=digest({k:v for k,v in value.items() if k!='bridge_id'}):
        raise ValueError('Changed multi-component bridge identity')
    a,b=value['pieces'];groups=value['source_groups'];merged=value['merged_groups']
    flat=[p for g in groups for p in g]
    if not a<b or len(value['arcs'])!=2 or [v['piece'] for v in value['arcs']]!=[a,b] or len(groups)<2 or any(not g or g!=sorted(set(g)) for g in groups) or len(flat)!=len(set(flat)):
        raise ValueError('Invalid multi-component source partition')
    if len(merged)!=2 or merged[0] not in groups or merged[1] not in groups or merged[0]==merged[1] or a not in merged[0] or b not in merged[1]:
        raise ValueError('Native contact does not cross its declared source components')
    if value['role'] not in {'merge','cycle'}:raise ValueError('Unknown native bridge graph role')
    if set(value['capture_binding'])!={'rotation_family','capture_profile_sha256'} or value['capture_binding']['rotation_family'] not in {'quarter_turn','continuous_rigid'}:
        raise ValueError('Native seam rotation seeds lack declared acquisition metadata')
    require_rigid(value['transform_b_to_a'])
    require_native_rotation(value['proposed_transform_b_to_a'],value['capture_binding'])


def verify_layout(problem,layout,config):
    from .outputs import native_probes
    from paper_reconstruction_contracts import map_points
    poses={p['piece_id']:np.asarray(p['native_to_component'],float) for p in layout['pieces']}
    joins=layout['selected_joins'];checked=0
    for join in joins:
        value=join.get('native_bridge',{})
        if value.get('policy')!=POLICY:continue
        if problem.capture_metadata!=value['capture_binding'] or value['capture_binding']['capture_profile_sha256']!=problem.constraint_profile['capture_profile_sha256']:
            raise ValueError('Native repair capture provenance differs from authenticated input')
        arcs=record_arcs(problem,value);check_arcs(problem,arcs,poses,config)
        a,b=value['pieces'];actual=relative(poses,a,b);proposed=np.asarray(value['proposed_transform_b_to_a'])
        probes=np.asarray(native_probes(problem.pieces[b]))
        if float(np.quantile(np.linalg.norm(map_points(probes,actual)-map_points(probes,proposed),axis=1),.95))>config.cycle_tolerance_px:
            raise ValueError('Native repair proposal and composed whole-piece pose disagree')
        length=min(v.length for v in arcs)
        if not (length>=96. and spread(arcs)>=.02):
            # A sliding seam needs a distinct selected physical cut witness.
            adjacency={p:set() for p in poses}
            for other in joins:
                if other['geometry_id']==join['geometry_id']:continue
                x,y=other['piece_a'],other['piece_b'];adjacency[x].add(y);adjacency[y].add(x)
            seen=set();todo=[a]
            while todo:
                p=todo.pop()
                if p in seen:continue
                seen.add(p);todo.extend(adjacency[p]-seen)
            if b not in seen:raise ValueError('Unconditioned native repair bridge lacks selected-cycle corroboration')
        checked+=1
    return checked
