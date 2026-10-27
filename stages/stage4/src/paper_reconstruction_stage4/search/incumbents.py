"""Bounded final-feasibility cache and protected, cold-generated incumbents."""
from collections import Counter, OrderedDict
import time
from ..active_support import verify_result_support
from ..outputs import serialize_layout
from .common import retain
from .execution import clone, state_key


class Incumbents:
    def __init__(self,search,deadline):
        self.search=search;self.config=search.config;self.deadline=deadline
        self.cache=OrderedDict();self.routes={};self.counts=Counter();self.seconds=0.
        self.pair_checks=0
        self.page_bank = None
        if search.problem.page_context is not None:
            context = search.problem.page_context
            if hasattr(context,'make_bank'):
                self.page_bank = context.make_bank(deadline)
            else:
                from ..page import PageBank
                self.page_bank = PageBank(context, deadline)

    def route(self,name,deadline):
        value=Route(self,name,deadline)
        self.routes[name]=value
        return value

    def available(self,route):
        return time.perf_counter()<min(self.deadline,route.deadline)

    def check(self,route,state):
        key=state_key(state)
        if key in self.cache:
            self.counts['cache_hits']+=1;self.cache.move_to_end(key)
            return self.cache[key]
        if route.checks>=self.config.joint_final_check_limit or not self.available(route):
            route.counts['unverified_abstentions']+=1
            return None
        started=time.perf_counter();route.checks+=1
        check=self.search.feasibility.state(state,final=True)
        if check['valid']:
            layout=serialize_layout(self.search.problem,state,0,check,{'attempted':False})
            try:
                verify_result_support(self.search.problem,{'layouts':[layout]},self.config)
            except ValueError as exc:
                check={'valid':False,'reason':'native_support:'+str(exc)}
        self.seconds+=time.perf_counter()-started
        value={'valid':check['valid'],'reason':check['reason']}
        if len(self.cache)>=min(self.config.maximum_cache_entries,512):self.cache.popitem(last=False)
        self.cache[key]=value
        route.counts['valid' if value['valid'] else 'rejected_'+value['reason']]+=1
        return value

    def report(self):
        return {'policy':self.config.joint_finalist_policy,'counts':dict(self.counts),
            'final_check_seconds':self.seconds,'cached_states':len(self.cache),
            'early_pair_checks':self.pair_checks,'early_pair_check_limit':self.config.joint_early_pair_checks,
            'check_limit_per_route':self.config.joint_final_check_limit,
            'incumbents_per_route':self.config.joint_incumbent_limit,
            'routes':{name:route.report() for name,route in self.routes.items()},
            'thresholds_changed':False,'initial_result_seeds':False,'evaluation_truth_opened':False}


class Route:
    def __init__(self,shared,name,deadline):
        self.shared=shared;self.name=name;self.deadline=deadline
        self.checks=0;self.counts=Counter();self.states=[];self.origins=[]

    @property
    def checked(self):return self.shared.config.joint_finalist_policy=='checked-incumbents.v1'

    def consider(self,state,origin):
        if self.shared.page_bank is not None:
            self.shared.page_bank.consider(state, self.name+':'+origin)
        result=self.shared.check(self,state)
        if result is not None and not result['valid'] and self.shared.page_bank is not None:
            self.shared.page_bank.states.pop(state_key(state),None)
            self.shared.page_bank.origins.pop(state_key(state),None)
        if result is None or not result['valid']:return result
        candidate=clone(state)
        candidate.score=self.shared.search.problem.structural.evaluate(candidate)['score']
        self.states=retain([*self.states,candidate],self.shared.config.joint_incumbent_limit)
        if len(self.origins)<24:
            self.origins.append({'origin':origin,'common_score':candidate.score,'selected_geometry_ids':list(candidate.selected)})
        return result

    def admit_complete(self,state):
        if self.shared.page_bank is not None:
            self.shared.page_bank.consider(state,self.name+':generated_complete')
        if not self.checked or len(state.components())!=1:return True
        result=self.consider(state,'early_complete')
        return result is None or result['valid']

    def check_pair(self,a,b,h,coarse):
        shared=self.shared
        if not self.checked or coarse.get('overlap',0.)<.75*shared.config.maximum_assembly_collision:
            return coarse
        if shared.pair_checks>=shared.config.joint_early_pair_checks or not shared.available(self):
            self.counts['early_pair_abstentions']+=1
            return coarse
        shared.pair_checks+=1
        check=shared.search.feasibility.pair(a,b,h,final=True)
        self.counts['early_pair_valid' if check['valid'] else 'early_pair_rejected']+=1
        return check

    def report(self):
        return {'native_state_checks':self.checks,'counts':dict(self.counts),
            'protected_incumbents':len(self.states),'incumbent_origins':self.origins}
