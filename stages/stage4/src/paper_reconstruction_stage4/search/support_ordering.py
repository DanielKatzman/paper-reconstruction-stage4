"""Bounded assigned-support priors; never change incoming poses or occurrences."""
from dataclasses import replace
import time
from ..support import contact_error


def ordering_priority(search, proto, edge):
    audit = search.ordering_audit
    counts = audit['counts']; counts['edges_considered'] += 1
    deadline = search.problem.deadline
    if deadline is not None and time.perf_counter() >= deadline-search.config.joint_finalize_seconds:
        counts['deadline_unscored_fallbacks'] += 1
        return max(-1.,min(1.,edge.weight))-search.config.join_cost
    assignments, reason, nodes = search.support.assign((edge.id,),proto.poses,search.edges)
    search.assignment_nodes += nodes; counts['assignment_nodes'] += nodes
    if assignments is None:
        counts['unassignable_edges'] += 1
        counts[reason] += 1
        # Keep this pose searchable: another component gauge/support context
        # might make it useful. The unassigned prior is explicitly a fallback.
        return search.problem.structural.evaluate(proto)['score']
    assigned = replace(proto,assignments=assignments)
    if hasattr(search.support,'bind'):
        search.support.bind(assigned)
    candidates = [assigned]
    seen = {tuple((a.piece,a.intervals) for a in search.support.arcs[assignments[edge.id]])}
    for occurrence in sorted(edge.occurrences,key=lambda o:o['occurrence_id']):
        if len(candidates) >= search.config.joint_ordering_occurrences:
            break
        oid = occurrence['occurrence_id']; arcs = search.support.arcs.get(oid)
        if arcs is None:
            continue
        key = tuple((a.piece,a.intervals) for a in arcs)
        if key in seen:
            continue
        seen.add(key)
        if contact_error(arcs,proto.poses) <= search.config.contact_tolerance_px:
            candidates.append(replace(proto,assignments={edge.id:oid},active_support={}))
    terms = [search.problem.structural.evaluate(s) for s in candidates]
    best = max(range(len(terms)),key=lambda i:(terms[i]['score'],-i))
    counts['assigned_edges'] += 1; counts['support_options_scored'] += len(terms)
    counts['continuation_observed_options'] += sum(bool(t.get('continuation_evidence',{}).get('joins')) for t in terms)
    if len(audit['examples']) < 16:
        state = candidates[best]
        audit['examples'].append({'geometry_id':edge.id,'support_options_scored':len(terms),
            'priority':terms[best]['score'],'occurrence_id':state.assignments[edge.id],
            'active_support_id':state.active_support.get(edge.id,{}).get('support_id'),
            'continuation_cost':terms[best].get('continuation_evidence',{}).get('cost',0.)})
    return terms[best]['score']
