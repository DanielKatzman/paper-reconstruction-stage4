"""Bridge predicted-mask connectivity without changing the raw Stage 1 output."""
import argparse
from pathlib import Path
import json
import shutil
import numpy as np
from PIL import Image
from scipy import ndimage
from common import inventory,save,sha

POLICY='bounded-predicted-mask-native-frame.v2'


def connected_mask(mask):
    labels,count=ndimage.label(mask)
    if count==1:return mask,0
    if count==0 or ndimage.label(mask,np.ones((3,3),bool))[1]!=1:
        raise ValueError('Mask requires review: it is empty or contains separate eight-connected fragments')
    areas=np.bincount(labels.ravel());areas[0]=0;largest=int(areas.argmax())
    removed=int(mask.sum()-areas[largest])
    if removed>64 or removed/max(1,int(mask.sum()))>1e-4:
        raise ValueError('Mask requires review: four-connectivity cleanup would remove a meaningful fragment')
    return labels==largest,removed


def prepare(source,output):
    source=Path(source).resolve();output=Path(output).resolve()
    if output.exists() or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Adapter output must be new and separate from its raw input')
    before=inventory(source);manifest=source/'segmentation_manifest.jsonl'
    rows=[json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines() if line.strip()]
    shutil.copytree(source,output);changes=[]
    for row in rows:
        if not row['accepted']:continue
        def artifact(key):
            path=(output/row[key]).resolve()
            if not path.is_relative_to(output):raise ValueError('Artifact escapes Stage 1 output')
            return path
        mask_path=artifact('piece_mask_path')
        old=np.asarray(Image.open(mask_path).convert('L'))>=128
        mask,removed=connected_mask(old)
        x,y,w,h=row['bbox_xywh']
        if old.shape!=(h,w):raise ValueError('Crop mask and declared bounding box differ')
        full_path=artifact('mask_path');full=np.asarray(Image.open(full_path).convert('L')).copy()>=128
        if not np.array_equal(full[y:y+h,x:x+w],old):raise ValueError('Full and cropped masks differ')
        full[y:y+h,x:x+w]=mask
        # Stage 2 traces pixel-cell boundaries; retain a real zero-mask margin
        # so half-pixel contours fit Stage 3's native image frame.
        yy,xx=np.nonzero(full);left=max(0,int(xx.min())-4);top=max(0,int(yy.min())-4)
        right=min(full.shape[1],int(xx.max())+5);bottom=min(full.shape[0],int(yy.max())+5)
        cropped=full[top:bottom,left:right]
        if cropped[0].any() or cropped[-1].any() or cropped[:,0].any() or cropped[:,-1].any():
            raise ValueError('Mask requires review: the piece touches the original capture frame')
        piece_path=artifact('piece_path');old_rgba=np.asarray(Image.open(piece_path).convert('RGBA'))
        if old_rgba.shape[:2]!=(h,w):raise ValueError('Piece image and declared bounding box differ')
        rgba=np.full((bottom-top,right-left,4),255,np.uint8)
        ix,iy=max(left,x),max(top,y);ex,ey=min(right,x+w),min(bottom,y+h)
        rgba[iy-top:ey-top,ix-left:ex-left]=old_rgba[iy-y:ey-y,ix-x:ex-x]
        rgba[:,:,3]=cropped.astype(np.uint8)*255
        Image.fromarray(full.astype(np.uint8)*255).save(full_path)
        Image.fromarray(cropped.astype(np.uint8)*255).save(mask_path)
        Image.fromarray(rgba).save(piece_path)
        prior=list(row['bbox_xywh']);row['bbox_xywh']=[left,top,right-left,bottom-top]
        row['downstream_mask_adjustment']={'policy':POLICY,'removed_pixels':removed,
            'raw_bbox_xywh':prior,'native_pixel_scale_changed':False,'raw_output_preserved':True,
            'target_zero_mask_margin_px':4,'native_crop_offset_xy':[x-left,y-top]}
        if 'diagnostics' in row:
            row['diagnostics']={**row['diagnostics'],'component_count':1,'area_fraction':float(full.mean())}
        changes.append({'capture_id':row['capture_id'],'removed_pixels':removed,'raw_bbox_xywh':prior,
            'bbox_xywh':row['bbox_xywh'],'adjusted_mask_sha256':sha(mask_path)})
    (output/'segmentation_manifest.jsonl').write_text(''.join(json.dumps(row,sort_keys=True)+'\n' for row in rows),encoding='utf-8')
    if inventory(source)!=before:raise ValueError('Raw Stage 1 output changed')
    receipt={'policy':POLICY,'source_manifest_sha256':sha(manifest),'source_files':before,'adjustments':changes,
        'maximum_removed_pixels':64,'maximum_removed_fraction':1e-4,'raw_input_unchanged':True,
        'resizing_or_perspective_correction':False,'result_manifest_sha256':sha(output/'segmentation_manifest.jsonl')}
    save(output/'mask_boundary_adapter.json',receipt)
    print(json.dumps({'status':'stage1_boundary_ready','adjusted_captures':len(changes),'removed_pixels':sum(x['removed_pixels'] for x in changes)}))
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();prepare(a.input,a.output)
