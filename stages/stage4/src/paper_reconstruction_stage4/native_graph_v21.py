"""Cooperative native growth whose frontiers survive scheduling and new modes."""
from collections import Counter, OrderedDict, defaultdict, deque
from dataclasses import dataclass, field
import time
import numpy as np
from .native_graph_v17 import Growth, offers, quality, identity, incremental_check
from .geometry import regauge
from .constraints import require_rigid
from .native_bridge import contour_arc, check_arcs
from .support import overlaps
from .repair_search import Node
from .search.execution import clone
from .component_closure import shape_measurement
from .page_geometry import state_record


@dataclass
class Frame:
    growth: Growth
    primary: bool
    tried: set = field(default_factory=set)
    choices: object = None
    revision: int = -1


class NativeFrontier:
    """A scheduling turn advances work; it never restarts or truncates a route.

    Beam pruning affects ordering only. Persistent DFS sessions reopen every
    seed and fitted mode. New-bank modes refresh live DFS choices and add new
    beam roots, without replaying old beam roots. All acceptance is delegated
    to the ordinary independent complete-page finalizer.
    """
    def __init__(self, context, basis, physical, on_complete, lane):
        if basis.selected or basis.assignments or basis.active_support or any(len(g)!=1 for g in basis.components()):
            raise ValueError('Independent native graph requires empty singletons')
        if any(not np.array_equal(h,np.eye(3)) for h in basis.poses.values()):
            raise ValueError('Independent native graph cannot inherit old poses')
        if not context.problem.capture_metadata or context.problem.capture_metadata.get('rotation_family') not in {'quarter_turn','continuous_rigid'}:
            raise ValueError('Native graph requires authenticated rigid capture metadata')
        self.context,self.basis,self.physical,self.on_complete,self.lane=context,basis,physical,on_complete,lane
        self.proposals={}; self.indexed=defaultdict(list); self.primary=defaultdict(list)
        self.seed_order=[]; self.seed_cursor=0; self.seed_primary=True
        self.batches=deque(); self.fast=None; self.dfs=None; self.sessions=deque()
        self.revision=0; self.answer=None; self.deepest=1; self.best=[]
        self.decoded={}; self.memo=OrderedDict(); self.final_memo=OrderedDict()
        self.counts=Counter(); self.rejections=Counter(); self.active_seconds=0.

    def add(self, proposals):
        fresh=[p for p in proposals if p['proposal_id'] not in self.proposals]
        if not fresh:
            return False
        self.proposals.update((p['proposal_id'],p) for p in fresh); self.revision+=1
        ordered=sorted(self.proposals.values(),key=lambda p:(not p['conditioned'],-quality(p),p['proposal_id']))
        self.indexed=defaultdict(list); self.primary=defaultdict(list)
        pairs=defaultdict(list)
        for p in ordered:
            pairs[tuple(p['pieces'])].append(p)
            for pid in p['pieces']:
                self.indexed[pid].append(p)
        from .seam_poses_v15 import retain
        for values in pairs.values():
            for p in retain(values):
                for pid in p['pieces']:
                    self.primary[pid].append(p)
        seeds=[]; rest=[]; seen=set()
        for p in sorted(fresh,key=lambda p:(not p['conditioned'],-quality(p),p['proposal_id'])):
            pair=tuple(p['pieces']); (rest if pair in seen else seeds).append(p); seen.add(pair)
        added=[*seeds,*rest]
        self.seed_order.extend(added)
        # Finish an existing positive ordering route with its original view.
        # A bank expansion gets new roots plus refreshed DFS choices; it must
        # not silently change offers halfway through an older beam frontier.
        self.batches.append((added,self.primary))
        # Exhausted generators may restart only for newly added roots. Live
        # generators/frontiers and their checked-offer sets remain intact.
        if self.dfs is None:
            self.dfs=self._backtracking()
        return True

    @property
    def fast_done(self):
        return self.fast is None and not self.batches

    @property
    def dfs_done(self):
        return self.dfs is None and self.seed_cursor>=len(self.seed_order) and not self.sessions

    def remember(self,cache,key):
        if key in cache:
            cache.move_to_end(key); return False
        cache[key]=True
        if len(cache)>self.context.config.maximum_cache_entries:
            cache.popitem(last=False)  # Eviction can repeat work, not discard branches.
        return True

    def choices(self,growth,primary=False):
        first=offers(growth,self.primary,self.context.config.cycle_tolerance_px)
        yield from first
        if not primary:
            used={r['proposal']['proposal_id'] for r in first}
            yield from (r for r in offers(growth,self.indexed,self.context.config.cycle_tolerance_px)
                        if r['proposal']['proposal_id'] not in used)

    def extend(self,growth,offer):
        self.counts['merge_checks']+=1
        p=offer['proposal']; a,b=p['pieces']; key=p['proposal_id']; state=growth.node.state
        if key not in self.decoded:
            try:
                h=np.asarray(p['transform_b_to_a']); require_rigid(h)
                if not np.allclose(h[:2,:2],np.rint(h[:2,:2]),atol=1e-7,rtol=0.):
                    raise ValueError('Undeclared graph rotation')
                arcs=tuple(contour_arc(self.context.problem,r) for r in p['arcs'])
                if [arc.piece for arc in arcs]!=[a,b]:
                    raise ValueError('Proposal arc pair differs')
                self.counts['native_proposal_replay_checks']+=1
                metrics=check_arcs(self.context.problem,arcs,{a:np.eye(3),b:h},self.context.config)
                self.decoded[key]=(arcs,metrics)
            except ValueError:
                self.decoded[key]=None
        if self.decoded[key] is None:
            self.rejections['native_proposal_replay']+=1; return None
        arcs,metrics=self.decoded[key]
        used=[arc for row in growth.node.contacts for arc in row['arcs']]
        occupied=any(overlaps(a,b,self.context.config) for a in arcs for b in used)
        if occupied:
            self.rejections['deferred_occupied_support']+=1
        candidate=clone(state); pid=offer['piece']; candidate.poses[pid]=offer['pose'].copy()
        members={x for x in state.poses if state.groups[x]==state.groups[growth.anchor]}|{pid}
        for x in members:
            candidate.groups[x]=min(members)
        candidate.poses=regauge(candidate.poses,members)
        valid,reason=incremental_check(self.context,candidate,pid,members,self.physical,self.rejections)
        if not valid:
            self.rejections[reason]+=1; return None
        contact={'pieces':[a,b],'arcs':arcs,'metrics':metrics,'proposal_id':key,
            'proposed_transform_b_to_a':p['transform_b_to_a'],'source_groups':state.components(),
            'merged_groups':[sorted(x for x in state.poses if state.groups[x]==state.groups[a]),
                             sorted(x for x in state.poses if state.groups[x]==state.groups[b])],
            'role':'merge','spread':p['shape_spread_ratio']}
        self.rejections['accepted']+=1
        value=offer['priority']-(.5*quality(p) if occupied else 0.)
        return Growth(Node(candidate,[*growth.node.contacts,contact],growth.node.value+value),growth.anchor,growth.seed)

    def seed(self,p):
        a,b=p['pieces']; initial=Growth(Node(clone(self.basis)),a,p['proposal_id'])
        return self.extend(initial,{'proposal':p,'piece':b,'pose':np.asarray(p['transform_b_to_a']),'priority':quality(p)})

    def consider(self,growth):
        state=growth.node.state; depth=sum(state.groups[p]==state.groups[growth.anchor] for p in state.poses)
        if depth>self.deepest:
            self.deepest=depth; self.best=[growth]
        elif depth==self.deepest and len(self.best)<3:
            self.best.append(growth)
        if depth!=len(state.poses):
            return False
        self.counts['complete_shape_candidates']+=1
        shape=shape_measurement(self.context,state,256)
        if not shape['shape_gate_pass']:
            self.rejections['complete_shape']+=1; return True
        if self.remember(self.final_memo,identity(growth)):
            self.counts['final_candidates_checked']+=1
            self.answer=self.on_complete(growth.node,shape)
        else:
            self.counts['duplicate_final_histories_avoided']+=1
        return True

    def _beam(self,roots,primary_index):
        nodes=[]
        for p in roots:
            if len(nodes)>=self.context.config.repair_beam//2:
                break
            growth=self.seed(p)
            if growth is not None:
                self.consider(growth); nodes.append(growth)
            yield
            if self.answer is not None:
                return
        while nodes:
            following=[]
            for growth in nodes:
                by_piece=defaultdict(list)
                for offer in offers(growth,primary_index,self.context.config.cycle_tolerance_px):
                    by_piece[offer['piece']].append(offer)
                expanded=0
                for values in by_piece.values():
                    accepted=0
                    for offer in values:
                        child=self.extend(growth,offer)
                        if child is not None:
                            if not self.consider(child):
                                following.append(child)
                            accepted+=1
                        yield
                        if self.answer is not None:
                            return
                        if accepted==2:
                            break
                    if accepted:
                        expanded+=1
                    if expanded==2:
                        break
            following.sort(key=lambda g:(-g.node.value,identity(g)))
            keep=[]; used=set(); families=set()
            for growth in following:
                if growth.seed not in families and len(keep)<self.context.config.repair_beam//2:
                    keep.append(growth); used.add(identity(growth)); families.add(growth.seed)
            for growth in following:
                key=identity(growth)
                if key not in used:
                    keep.append(growth); used.add(key)
                if len(keep)>=self.context.config.repair_beam:
                    break
            nodes=keep
            yield

    def _backtracking(self):
        while self.sessions or self.seed_cursor<len(self.seed_order):
            while len(self.sessions)<self.context.config.repair_beam and self.seed_cursor<len(self.seed_order):
                p=self.seed_order[self.seed_cursor]; primary=self.seed_primary
                self.seed_primary=not primary
                if not primary:
                    self.seed_cursor+=1
                self.counts['backtracking_seed_sessions_started']+=1
                growth=self.seed(p)
                if growth is not None and not self.consider(growth):
                    self.sessions.append([Frame(growth,primary)])
                yield
                if self.answer is not None:
                    return
            if not self.sessions:
                continue
            stack=self.sessions.popleft(); frame=stack[-1]
            if frame.revision!=self.revision:
                frame.choices=self.choices(frame.growth,frame.primary); frame.revision=self.revision
            offer=next(frame.choices,None)
            while offer is not None and offer['proposal']['proposal_id'] in frame.tried:
                offer=next(frame.choices,None)
            if offer is None:
                stack.pop(); self.counts['backtracks']+=1
            else:
                frame.tried.add(offer['proposal']['proposal_id'])
                child=self.extend(frame.growth,offer)
                if child is not None and self.remember(self.memo,(frame.primary,identity(child))):
                    if not self.consider(child):
                        stack.append(Frame(child,frame.primary))
            if stack:
                self.sessions.append(stack)
            yield
            if self.answer is not None:
                return

    def step(self,kind):
        started=time.perf_counter()
        try:
            if kind=='beam':
                if self.fast is None and self.batches:
                    roots,primary_index=self.batches.popleft()
                    self.fast=self._beam(roots,primary_index)
                if self.fast is None:
                    return False
                try:
                    next(self.fast)
                except StopIteration:
                    self.fast=None
                return True
            if self.dfs is None:
                return False
            try:
                next(self.dfs)
            except StopIteration:
                self.dfs=None
            return True
        finally:
            self.active_seconds+=time.perf_counter()-started

    def audit(self):
        return {**dict(self.counts),'rejections':dict(self.rejections),'growth_lane':self.lane,
            'completed_depth':self.deepest,'required_piece_depth':len(self.basis.poses),
            'elapsed_seconds':self.active_seconds,'bank_revision':self.revision,
            'beam_batches_pending':len(self.batches)+int(self.fast is not None),
            'backtracking_sessions_pending':len(self.sessions),'seed_cursor':self.seed_cursor,
            'seed_count':len(self.seed_order),'proposal_graph_exhausted':self.fast_done and self.dfs_done,
            'observed_growth_states':[{'state':state_record(g.node.state),'seed_proposal_id':g.seed,
                'value':g.node.value,'placement_proposal_ids':[r['proposal_id'] for r in g.node.contacts],
                'component_sizes':sorted(map(len,g.node.state.components()),reverse=True),'accepted_assembly':False}
                for g in self.best]}
