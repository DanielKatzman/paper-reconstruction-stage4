"""Bounded native colour/ink continuity for ambiguous, text-weak bridges.

The cue measures the supplied proposal and assigned native support, independently
of component gauges. No fitted transform, reference, OCR or page shape is used.
"""
from collections import Counter,OrderedDict
import time
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion,map_coordinates
from scipy.spatial import cKDTree
from paper_reconstruction_contracts import map_points


def image_profile(piece, maximum):
    with Image.open(piece.image_path) as image:
        image=image.convert('RGB');factor=min(1.,maximum/max(image.size))
        size=tuple(max(1,round(v*factor)) for v in image.size)
        native=np.asarray(image,np.uint8).copy();native[~piece.mask]=0
        rgb=np.asarray(Image.fromarray(native).resize(size,Image.Resampling.BILINEAR),np.uint8)
    alpha=np.asarray(Image.fromarray(piece.mask.astype(np.uint8)*255).resize(size,Image.Resampling.BILINEAR),np.uint8)
    mask=alpha>=254
    interior=binary_erosion(mask,iterations=3)
    if interior.sum()<64:interior=mask
    values=rgb[interior].astype(float)/255.
    white=np.quantile(values,.90,axis=0) if len(values) else np.ones(3)
    # A nearly monochrome coloured/dark fragment cannot provide a dependable
    # paper white point. Abstain instead of stretching its range into contrast.
    reliable=len(values)>=64 and white.min()>=.45 and white.max()-white.min()<=.20
    return {'rgb':rgb,'alpha':alpha,'white':white,'reliable':bool(reliable),
        'scale_xy':np.array([size[0]/piece.shape[1],size[1]/piece.shape[0]])}


def sample(profile,points):
    xy=(points+.5)*profile['scale_xy']-.5
    alpha=map_coordinates(profile['alpha'],[xy[:,1],xy[:,0]],order=1,
        mode='constant',cval=0.,output=np.float32)/255.
    rgb=np.column_stack([map_coordinates(profile['rgb'][:,:,channel],[xy[:,1],xy[:,0]],
        order=1,mode='constant',cval=0.,output=np.float32)/255. for channel in range(3)])
    # Preserve white as white on blank pieces. Per-fragment dark-point/range
    # stretching would falsely disagree with text-bearing pieces.
    rgb=rgb/np.maximum(alpha[:,None],1e-6)
    return np.clip(rgb/np.maximum(profile['white'],.2),0.,1.),alpha>=.75


def inside(piece,points):
    xy=np.rint(points).astype(np.int64)
    valid=(xy[:,0]>=0)&(xy[:,1]>=0)&(xy[:,0]<piece.shape[1])&(xy[:,1]<piece.shape[0])
    mask=np.zeros(len(points),bool);mask[valid]=piece.mask[xy[valid,1],xy[valid,0]]
    return mask


def band_information(values):
    """Visible variation along a native seam, not a placement certificate."""
    values=np.asarray(values,float)
    if values.size==0:return 0.
    # Constant white, coloured or black bands carry no along-seam location.
    contrast=np.quantile(values,.90,axis=0)-np.quantile(values,.10,axis=0)
    return float(np.clip(np.max(contrast)/.20,0.,1.))


class Continuation:
    def __init__(self,scorer):
        self.scorer=scorer;self.profiles={};self.cache=OrderedDict();self.counts=Counter();self.seconds=Counter()

    def profile(self,pid):
        if pid not in self.profiles:
            started=time.perf_counter()
            self.profiles[pid]=image_profile(self.scorer.problem.pieces[pid],self.scorer.config.continuation_image_side)
            self.seconds['image_setup']+=time.perf_counter()-started
        return self.profiles[pid]

    def measurement(self,state,eid):
        scorer=self.scorer;edge=scorer.edges[eid];oid=state.assignments[eid]
        active=state.active_support.get(eid,{}).get('support_id')
        key=(eid,oid,active)
        if key in self.cache:
            self.counts['cache_hits']+=1;self.cache.move_to_end(key)
            return self.cache[key]
        self.counts['cache_misses']+=1;started=time.perf_counter()
        arcs=(scorer.support.arcs_for(state,eid) if hasattr(scorer.support,'arcs_for') else scorer.support.arcs[oid])
        profiles={arc.piece:self.profile(arc.piece) for arc in arcs}
        errors=[];directions=[];information=[]
        from .config import AMBIGUITY_SELECTION
        measure_information=scorer.config.page_selection_policy==AMBIGUITY_SELECTION
        if all(p['reliable'] for p in profiles.values()):
            for first,second in (arcs,arcs[::-1]):
                pa,pb=scorer.problem.pieces[first.piece],scorer.problem.pieces[second.piece]
                h=np.linalg.inv(edge.transform) if first.piece==edge.a else edge.transform
                points=first.points[np.linspace(0,len(first.points)-1,scorer.config.continuation_samples).astype(int)]
                mapped=map_points(points,h);distance,index=cKDTree(second.points).query(mapped);target=second.points[index]
                na=pa.contour['inward_normal_xy'][scorer.normals[first.piece].query(points)[1]]
                nb=pb.contour['inward_normal_xy'][scorer.normals[second.piece].query(target)[1]]
                magnification=np.linalg.norm(map_points(points+na,h)-mapped,axis=1)
                band=[];valids=[];colours_a=[];colours_b=[]
                for depth in (2.,4.,8.,16.):
                    a=points+na*depth;b=target+nb*(depth*magnification)[:,None]
                    ca,visible_a=sample(profiles[first.piece],a);cb,visible_b=sample(profiles[second.piece],b)
                    valid=(distance<=scorer.config.contact_tolerance_px)&inside(pa,a)&inside(pb,b)&visible_a&visible_b
                    band.append(np.mean(np.abs(ca-cb),axis=1))
                    valids.append(valid)
                    if measure_information:colours_a.append(ca);colours_b.append(cb)
                band=np.asarray(band);valid=np.asarray(valids).all(axis=0)
                coverage=float(valid.mean());directions.append(coverage)
                if valid.any():errors.extend(np.quantile(band[:,valid],.25,axis=0).tolist())
                if measure_information and valid.any():
                    # Both sides must vary; a text/blank comparison cannot
                    # establish positive positional support for a sliding seam.
                    information.append(min(max(band_information(v[valid]) for v in colours_a),
                                           max(band_information(v[valid]) for v in colours_b)))
        coverage=min(directions,default=0.)
        usable=bool(errors) and coverage>=scorer.config.continuation_min_coverage
        value={'mismatch':float(np.mean(errors)) if usable else None,'coverage':coverage,
            'samples_used':len(errors),'normalization_reliable':all(p['reliable'] for p in profiles.values()),
            'reason':'observed_native_band_discontinuity' if usable else 'insufficient_reliable_visible_bands',
            'geometry_source':'supplied native proposal; assigned raw/active supports',
            'poses_refitted':False,'bidirectional':True}
        if measure_information:
            info=min(information,default=0.) if usable else 0.
            value.update(appearance_information=info,positive_placement_evidence=False,
                information_semantics='Along-seam variation on both visible sides; variation alone does not resolve competing displacement basins',
                appearance_evidence_status='informative_local_mismatch_only' if info>=.25 else 'uninformative_for_placement')
        if len(self.cache)>=scorer.config.maximum_cache_entries:self.cache.popitem(last=False)
        self.cache[key]=value;self.seconds['measurements_including_image_setup']+=time.perf_counter()-started
        return value

    def evaluate(self,state,attachment):
        rows=[];total=0.;scorer=self.scorer
        all_axes=scorer.config.objective_version == 'observable-joint.v8-continuation'
        policy='all-axis-native-continuation.v2' if all_axes else 'text-weak-native-continuation.v1'
        if scorer.config.continuation_weight==0.:
            return {'policy':policy,'cost':0.,'joins':[],
                'reason':'zero_weight','confidence_calibrated':False,'hard_rejection':False,'evaluation_truth_opened':False}
        for bridge in attachment['bridges']:
            # Existing independent selected cycles already bypass this bridge
            # list. Strong text cues keep their established scoring semantics.
            if not all_axes and bridge['axis_confidence']>=.5:continue
            eid=bridge['geometry_id']
            if eid not in state.assignments:continue
            value=self.measurement(state,eid)
            discount=1.-scorer.config.attachment_corroboration_discount if bridge['assignable_corroboration'] else 1.
            cost=0. if value['mismatch'] is None else scorer.config.continuation_weight*discount*value['mismatch']
            rows.append({'geometry_id':eid,'pieces':bridge['pieces'],'axis_confidence':bridge['axis_confidence'],
                'corroboration_discount_factor':discount,'cost':cost,**value});total+=cost
        return {'policy':policy,'cost':total,'joins':rows,
            'confidence_calibrated':False,'hard_rejection':False,'evaluation_truth_opened':False}

    def report(self):
        return {'counts':dict(self.counts),'phase_seconds':dict(self.seconds),
            'profiled_pieces':len(self.profiles),'cached_measurements':len(self.cache),
            'cache_capacity':self.scorer.config.maximum_cache_entries,'profiles_dtype':'uint8 premultiplied RGB and coverage',
            'maximum_image_side':self.scorer.config.continuation_image_side,
            'timing_scope':'Nested cumulative measurements; image setup is included in measurement time'}


def evaluate(scorer,state,attachment):
    if not hasattr(scorer,'continuation_context'):scorer.continuation_context=Continuation(scorer)
    return scorer.continuation_context.evaluate(state,attachment)
