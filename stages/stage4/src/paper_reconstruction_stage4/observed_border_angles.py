"""Rotation hypotheses from long observed contour spans, never connecting evidence.

The complete-rectangle contract makes external borders parallel/perpendicular.
A straight span may instead be a cut: every resulting pose still needs native
seam, support, collision and page verification. NumPy/SciPy only, CPU portable.
"""
import numpy as np


def _simplify(points, indices, epsilon):
    keep={0,len(indices)-1};pending=[(0,len(indices)-1)]
    while pending:
        first,last=pending.pop()
        if last-first<2:continue
        a,b=points[indices[first]],points[indices[last]];v=b-a
        xy=points[indices[first+1:last]]
        t=np.clip((xy-a)@v/max(float(v@v),1e-12),0.,1.)
        distance=np.linalg.norm(xy-a-t[:,None]*v,axis=1)
        j=int(np.argmax(distance))
        if distance[j]>epsilon:
            middle=first+1+j;keep.add(middle);pending.extend([(first,middle),(middle,last)])
    return [indices[i] for i in sorted(keep)]


def lines(points, contact_tolerance=12.):
    """Closed-contour simplification followed by full-span least-squares fitting."""
    xy=np.asarray(points,float)
    if xy.ndim!=2 or xy.shape[1]!=2 or len(xy)<4 or not np.isfinite(xy).all():
        raise ValueError('Invalid observed contour')
    # Split the closed contour at a distant vertex, avoiding a zero chord.
    far=int(np.argmax(np.sum((xy-xy[0])**2,axis=1)))
    if far==0:return []
    epsilon=contact_tolerance/2.
    vertices=(_simplify(xy,list(range(far+1)),epsilon)[:-1]+
        _simplify(xy,[*range(far,len(xy)),0],epsilon)[:-1])
    result=[]
    for first,last in zip(vertices,vertices[1:]+vertices[:1]):
        length=float(np.linalg.norm(xy[last]-xy[first]))
        if length<128.:continue
        segment=xy[np.arange(first,first+(last-first)%len(xy)+1)%len(xy)]
        centered=segment-segment.mean(axis=0)
        _,_,axes=np.linalg.svd(centered,full_matrices=False)
        rms=float(np.sqrt(np.mean((centered@axes[1])**2)))
        if rms>contact_tolerance/4.:continue
        result.append({'angle':float(np.mod(np.arctan2(axes[0,1],axes[0,0]),np.pi)),
            'length_native_px':length,'line_rms_native_px':rms,'first_index':first,'last_index':last})
    result.sort(key=lambda r:(-r['length_native_px'],r['first_index']))
    return [r for r in result if r['length_native_px']>=max(256.,result[0]['length_native_px']*.5)] if result else []


def angle_seeds(first,second):
    result=[];step=np.deg2rad(.25)
    for a in first:
        for b in second:
            for quarter in range(4):
                angle=a['angle']-b['angle']+quarter*np.pi/2.
                if any(abs(np.arctan2(np.sin(angle-r['angle']),np.cos(angle-r['angle'])))<step for r in result):continue
                result.append({'angle':angle,'source_geometry_id':None,
                    'observed_border_seed':{'first_span':a,'second_span':b,'parallel_perpendicular_basin':quarter}})
    return result
