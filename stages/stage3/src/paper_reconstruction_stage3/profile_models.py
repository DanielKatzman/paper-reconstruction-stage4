"""Alpha runtime extracted from reviewed v0.9.31; no research coordinator."""
from __future__ import annotations
from contextlib import nullcontext
from dataclasses import replace
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .contracts import require

def channels(bundle):
    return 10+(4 if "+P" in bundle else 0)+(4 if "A" in bundle else 0)


class ResidualConv(nn.Module):
    def __init__(self,width,dilation):
        super().__init__()
        self.net=nn.Sequential(nn.Conv1d(width,width,5,padding=2*dilation,dilation=dilation),
            nn.GroupNorm(8,width),nn.GELU(),nn.Conv1d(width,width,3,padding=dilation,dilation=dilation),nn.GroupNorm(8,width))
    def forward(self,x):return F.gelu(x+self.net(x))


class LocalPointBlock(nn.Module):
    """Adapted 2-D geometry attention: kNN neighborhoods, relative-position bias."""
    def __init__(self,width,k=16):
        super().__init__();self.k=k
        self.q=nn.Linear(width,width);self.kv=nn.Linear(width,width*2)
        self.position=nn.Sequential(nn.Linear(2,width),nn.GELU(),nn.Linear(width,width))
        self.norm=nn.LayerNorm(width);self.ff=nn.Sequential(nn.Linear(width,width*4),nn.GELU(),nn.Linear(width*4,width))
    def forward(self,x,xy):
        # Float32 distances also work on Pascal without FP16 cdist kernels.
        neighbors=torch.cdist(xy.float(),xy.float()).topk(min(self.k,x.shape[1]),largest=False).indices
        batch=torch.arange(x.shape[0],device=x.device)[:,None,None]
        key,value=self.kv(x).chunk(2,-1)
        relative=self.position(xy[:,:,None]-xy[batch,neighbors])
        score=((self.q(x)[:,:,None]-key[batch,neighbors]+relative).float().square().mean(-1)*-1).softmax(-1)
        out=(score[...,None]*(value[batch,neighbors]+relative)).sum(2)
        x=self.norm(x+out);return x+self.ff(x)


class SequenceEncoder(nn.Module):
    def __init__(self,spec,bundle,checkpointed=False):
        super().__init__();self.kind=spec.name.split("_")[0];self.checkpointed=checkpointed
        self.stem=nn.Linear(channels(bundle),spec.width);self.width=spec.width
        self.blocks=nn.ModuleList()
        if self.kind in {"tcn","hybrid"}:
            self.blocks.extend(ResidualConv(spec.width,2**(i%4)) for i in range(spec.layers if self.kind=="tcn" else 2))
        if self.kind in {"transformer","hybrid"}:
            self.attention=nn.ModuleList(nn.TransformerEncoderLayer(spec.width,spec.heads,spec.width*4,
                dropout=.1,batch_first=True,norm_first=True) for _ in range(spec.layers))
        elif spec.name=="point_local_attention":
            self.point=nn.ModuleList(LocalPointBlock(spec.width) for _ in range(spec.layers))
        elif spec.name=="pointnet":
            self.point=nn.ModuleList(nn.Sequential(nn.Linear(spec.width,spec.width*2),nn.GELU(),nn.Linear(spec.width*2,spec.width),nn.LayerNorm(spec.width)) for _ in range(spec.layers))
        self.norm=nn.LayerNorm(spec.width)

    def call(self,layer,*args):
        return checkpoint(layer,*args,use_reentrant=False) if self.checkpointed and self.training else layer(*args)

    def forward(self,sequence):
        x=self.stem(sequence)
        if self.blocks:
            x=x.transpose(1,2)
            for layer in self.blocks:x=self.call(layer,x)
            x=x.transpose(1,2)
        if hasattr(self,"attention"):
            # Observable normalized arc position; no owner/canonical coordinates.
            position=torch.linspace(0,1,x.shape[1],device=x.device,dtype=x.dtype)[None,:,None]
            frequencies=torch.arange(x.shape[2],device=x.device,dtype=x.dtype)[None,None,:]+1
            x=x+.1*torch.sin(position*frequencies)
            for layer in self.attention:x=self.call(layer,x)
        if hasattr(self,"point"):
            for layer in self.point:
                x=self.call(layer,x,sequence[...,:2]) if isinstance(layer,LocalPointBlock) else x+self.call(layer,x)
        return self.norm(x)


class CrossBlock(nn.Module):
    def __init__(self,width,heads):
        super().__init__();self.attention=nn.MultiheadAttention(width,heads,dropout=.1,batch_first=True)
        self.norm=nn.LayerNorm(width);self.ff=nn.Sequential(nn.Linear(width,width*4),nn.GELU(),nn.Linear(width*4,width))
    def forward(self,a,b):
        x=self.norm(a+self.attention(a,b,b,need_weights=False)[0]);return x+self.ff(x)


class PairModel(nn.Module):
    def __init__(self,spec,config,trial):
        super().__init__();self.kind=spec.family;self.checkpointed=trial["mode"]=="checkpointed";self.receipt={"pretrained":False,"initialization":"seeded scratch"}
        self.fp32_affinity=bool(trial.get("numerics_protocol"))
        from dataclasses import replace
        self.geometry=SequenceEncoder(replace(spec,name="tcn_small",width=128,layers=2),trial["bundle"],self.checkpointed) if self.kind!="P3" else None
        if self.kind=="P3":
            self.stem=nn.Linear(channels(trial["bundle"]),spec.width)
            self.blocks=nn.ModuleList(CrossBlock(spec.width,spec.heads) for _ in range(spec.layers))
            self.dustbin=nn.Parameter(torch.tensor(0.));dim=spec.width*4
        elif self.kind in {"P2","P4"}:
            self.cnn=nn.Sequential(nn.Conv2d(8,64,3,padding=1),nn.GroupNorm(8,64),nn.GELU(),
                nn.Conv2d(64,128,3,stride=2,padding=1),nn.GroupNorm(8,128),nn.GELU(),
                nn.Conv2d(128,256,3,stride=2,padding=1),nn.GroupNorm(8,256),nn.GELU(),nn.AdaptiveAvgPool2d(1))
            dim=256+128*4+8
        else:dim=128*4+8
        self.score=nn.Sequential(nn.Linear(dim,256),nn.GELU(),nn.Linear(256,1))

    def call(self,layer,*args):
        return checkpoint(layer,*args,use_reentrant=False) if self.checkpointed and self.training else layer(*args)

    def forward(self,a,b,evidence,joined=None):
        out={}
        if self.kind=="P3":
            x,y=self.stem(a["sequence"]),self.stem(b["sequence"])
            for layer in self.blocks:
                xx=self.call(layer,x,y);y=self.call(layer,y,x);x=xx
            with torch.autocast(device_type=x.device.type,enabled=False) if self.fp32_affinity else nullcontext():
                affinity=torch.bmm(x.float(),y.float().transpose(1,2))/x.shape[-1]**.5
            out["row_logits"]=torch.cat((affinity,self.dustbin.expand(*affinity.shape[:2],1)),2)
            out["column_logits"]=torch.cat((affinity.transpose(1,2),self.dustbin.expand(affinity.shape[0],affinity.shape[2],1)),2)
            # Shared symmetric scorer: swapping the pair leaves compatibility unchanged in eval.
            px=torch.cat((x.mean(1),x.amax(1)),1);py=torch.cat((y.mean(1),y.amax(1)),1)
            features=torch.cat((torch.abs(px-py),px*py),1)
        else:
            x,y=self.geometry(a["sequence"]),self.geometry(b["sequence"])
            px=torch.cat((x.mean(1),x.amax(1)),1);py=torch.cat((y.mean(1),y.amax(1)),1)
            features=torch.cat((torch.abs(px-py),px*py,evidence),1)
            if self.kind in {"P2","P4"}:
                image=torch.cat((a["strip"],torch.flip(b["strip"],[2])),1) if self.kind=="P2" else joined
                require(image is not None,"P4 needs observable proposed strip alignment")
                # Average both side orders; no arbitrary query side controls score.
                other=torch.cat((image[:,4:],image[:,:4]),1)
                vision=(self.call(self.cnn,image)+self.call(self.cnn,other)).flatten(1)/2
                features=torch.cat((features,vision),1)
        out["compatibility"]=self.score(features).squeeze(-1);return out
