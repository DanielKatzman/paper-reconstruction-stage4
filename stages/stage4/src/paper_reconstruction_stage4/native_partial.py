"""Verified partial output and repair of earlier attachments in stalled groups."""
from dataclasses import replace
from collections import Counter
import numpy as np
from .native_bridge import check_arcs,edge_from_record
from .native_bridge_v2 import make_record
from .active_support import verify_result_support
from .outputs import serialize_layout
from .page_geometry import fingerprint
from .search.execution import clone
from .cut_repair_v21 import cut_options,moved_node


def evaluate_contacts(context,source,node,contacts,physical):
    """Use the ordinary support verifier; a partial outline is never a page pass."""
    state=clone(node.state);used=[];edges=[]
    if source.selected:raise ValueError('Native partial publication requires independent singleton source')
    if not contacts:return None,{'disposition':'rejected_partial','reason':'no_supported_contacts'}
    try:
        for contact in contacts:
            a,b=contact['pieces']
            if state.groups[a]!=state.groups[b]:raise ValueError('Partial contact crosses unplaced components')
            check_arcs(context.problem,contact['arcs'],state.poses,context.config,used)
            used.extend(contact['arcs'])
            record=make_record(context.problem,state,contact,fingerprint(source))
            edge,arcs=edge_from_record(context.problem,record,context.config)
            state.selected=(*state.selected,edge.id);state.assignments[edge.id]=edge.occurrences[0]['occurrence_id']
            if contact['role']=='merge':state.tree=(*state.tree,edge.id)
            edges.append((edge,arcs))
        check=physical.state(state,final=True)
        if not check['valid']:return None,{'disposition':'rejected_partial','reason':check['reason']}
        temporary=replace(context.problem,edges=[*context.problem.edges,*[e for e,_ in edges if e.id not in context.edges]])
        verify_result_support(temporary,{'layouts':[serialize_layout(temporary,state,0,check,{'attempted':False})]},context.config)
    except ValueError as exc:
        return None,{'disposition':'rejected_partial_support','reason':str(exc)}
    for edge,arcs in edges:
        if edge.id not in context.edges:context.problem.edges.append(edge);context.edges[edge.id]=edge
        context.support.arcs[edge.occurrences[0]['occurrence_id']]=arcs
    context.cache.clear();context.ambiguity_cache.clear()
    scorer=context.scorer();scorer.edges.update(context.edges)
    state.score=float(scorer.evaluate(state)['score'])
    return state,{'disposition':'verified_partial','candidate_id':fingerprint(state),
        'component_sizes':sorted(map(len,state.components()),reverse=True),'complete_page_claimed':False}


def component_cuts(node,scope):
    """Every bridge of the placed tree is eligible; free pieces stay separate."""
    scope=set(scope)
    ordered=sorted(node.contacts,key=lambda r:(r.get('spread',0.),r['proposal_id']))
    for removed in ordered:
        if removed['role']!='merge':continue
        a,b=removed['pieces'];adj={p:set() for p in scope}
        for row in node.contacts:
            if row is removed:continue
            x,y=row['pieces']
            if x in scope and y in scope:adj[x].add(y);adj[y].add(x)
        seen=set();todo=[a]
        while todo:
            p=todo.pop()
            if p in seen:continue
            seen.add(p);todo.extend(adj[p]-seen)
        if b in seen:continue
        sides=[tuple(sorted(seen)),tuple(sorted(scope-seen))]
        yield min(sides,key=lambda x:(len(x),x))


class PartialRepair:
    def __init__(self,growth):
        self.growth=growth;self.revision=-1;self.iterator=None;self.exhausted=False
        self.tried=set();self.counts=Counter()

    def update(self,context):
        if self.revision==context.proposal_revision:return
        self.revision=context.proposal_revision;self.exhausted=False
        scope=max(self.growth.node.state.components(),key=len)
        def options():
            for moving in component_cuts(self.growth.node,scope):
                rows=[r for r in context.generated if all(p in scope for p in r['pieces'])]
                for option in cut_options(self.growth.node,moving,rows,context.config.cycle_tolerance_px,
                        context.problem.capture_metadata['rotation_family']):
                    key=(moving,option['proposal']['proposal_id'])
                    if key not in self.tried:
                        self.tried.add(key);yield moving,option,scope
        self.iterator=options()

    def step(self,context,physical):
        self.update(context);value=next(self.iterator,None)
        if value is None:self.exhausted=True;return None,False
        moving,option,scope=value;self.counts['alternatives_examined']+=1
        try:
            node=moved_node(context,context.native_basis,self.growth.node,moving,option,scope=scope)
            if not context.component_bounds(node.state)['valid']:
                self.counts['component_area_bound']+=1;return None,True
            verdict=physical.state(node.state,final=False)
            if not verdict['valid']:
                self.counts[verdict['reason']]+=1;return None,True
            self.counts['physically_valid_repaired_partials']+=1
            return node,True
        except ValueError as exc:
            self.counts[getattr(exc,'reason','native_contact_replay')]+=1
            return None,True
