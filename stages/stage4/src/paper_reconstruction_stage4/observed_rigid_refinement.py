"""Local SE2 fitting from observed opposing contours and common page borders.

This yields numerical proposals, never accepted layouts. The caller must rebuild
all support at the new poses and independently replay every published contact.
"""
import time
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import least_squares
from .geometry import map_points
from .page import enclosing_rectangle
from .native_bridge import runs


def proposals(context,node,deadline):
    p=context.problem;poses={pid:h.copy() for pid,h in node.state.poses.items()};ids=sorted(poses)
    anchor=max(ids,key=lambda pid:context.areas[pid]);free=[pid for pid in ids if pid!=anchor]
    xy={pid:np.asarray(p.pieces[pid].contour['points_xy'],float) for pid in ids}
    normals={pid:np.asarray(p.pieces[pid].contour['inward_normal_xy'],float) for pid in ids}
    centres={pid:xy[pid].mean(0) for pid in ids};pairs=[r['pieces'] for r in node.contacts]
    # A numerical iteration cap is a local proposal bound, not a solver stop rule.
    for iteration in range(10):
        if time.perf_counter()>=deadline:return
        world={pid:map_points(xy[pid],poses[pid]) for pid in ids}
        nw={pid:normals[pid]@poses[pid][:2,:2].T for pid in ids};matches=[]
        for a,b in pairs:
            for first,second in [(a,b),(b,a)]:
                distance,near=cKDTree(world[second]).query(world[first])
                cosine=np.sum(nw[first]*nw[second][near],axis=1)
                ix=np.flatnonzero((distance<2.5*context.config.contact_tolerance_px)&(cosine<-.65))
                if len(ix)<20:continue
                if len(ix)>300:ix=ix[np.linspace(0,len(ix)-1,300).astype(int)]
                matches.append((first,second,xy[first][ix],xy[second][near[ix]]))
        _,axes,lo,hi=enclosing_rectangle(np.vstack(list(world.values())))
        theta=float(np.arctan2(axes[1,0],axes[0,0]));edges=[]
        for pid in ids:
            frame=world[pid]@axes;nn=nw[pid]@axes
            for axis,side in [(0,0),(0,1),(1,0),(1,1)]:
                target=(lo if side==0 else hi)[axis]
                good=(np.abs(frame[:,axis]-target)<3.*context.config.contact_tolerance_px)&(np.abs(nn[:,axis])>.90)
                selected=[]
                for first,last in runs(good):
                    ix=np.arange(first,last+1)%len(good)
                    if len(ix)>=40:selected.extend(ix.tolist())
                if len(selected)<40:continue
                selected=np.asarray(selected)
                if len(selected)>180:selected=selected[np.linspace(0,len(selected)-1,180).astype(int)]
                edges.append((pid,axis,side,xy[pid][selected]))
        if not matches:return
        def unpack(z):
            out=dict(poses)
            for i,pid in enumerate(free):
                dx,dy,angle=z[3*i:3*i+3];angle/=1000.
                cc,ss=np.cos(angle),np.sin(angle);rotation=np.array([[cc,-ss],[ss,cc]])
                centre=map_points(centres[pid][None],poses[pid])[0]
                h=np.eye(3);h[:2,:2]=rotation@poses[pid][:2,:2]
                h[:2,2]=centre+[dx,dy]-h[:2,:2]@centres[pid];out[pid]=h
            last=z[3*len(free):];angle=theta+last[4]/1000.
            cc,ss=np.cos(angle),np.sin(angle)
            return out,np.array([[cc,-ss],[ss,cc]]),lo+last[:2],hi+last[2:4]
        def residual(z):
            if time.perf_counter()>=deadline:raise TimeoutError('Observed rigid fitting deadline')
            pp,ax,l,h=unpack(z);parts=[]
            for a,b,xa,xb in matches:
                parts.append(((map_points(xa,pp[a])-map_points(xb,pp[b]))*np.sqrt(180./len(xa))).ravel())
            for pid,axis,side,points in edges:
                parts.append(((map_points(points,pp[pid])@ax)[:,axis]-(l if side==0 else h)[axis])*np.sqrt(120./len(points)))
            return np.concatenate(parts)
        z0=np.zeros(3*len(free)+5)
        try:
            before=float(np.mean(residual(z0)**2))
            fitted=least_squares(residual,z0,loss='soft_l1',f_scale=2.,max_nfev=60,ftol=1e-6,xtol=1e-6,gtol=1e-5)
        except TimeoutError:return
        after=float(np.mean(fitted.fun**2));candidate=unpack(fitted.x)[0]
        yield candidate,{'iteration':iteration,'objective_before':before,'objective_after':after,
            'optimizer_success':bool(fitted.success),'seam_blocks':len(matches),'border_blocks':len(edges),
            'anchor':anchor,'relative_scale':1.,'reference_used':False}
        if not np.isfinite(after) or after>=before:return
        poses=candidate
        if np.max(np.abs(fitted.x))<.03:return
