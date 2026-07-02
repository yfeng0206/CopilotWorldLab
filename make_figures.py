# -*- coding: utf-8 -*-
"""Lean schematic figures for the Copilot Automation Lab proposal (generated, not copied)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
import os
OUT = os.path.dirname(os.path.abspath(__file__))
BLUE="#2f5d8a"; LEARN="#b5651d"; CLASS="#3a7d44"; AMBER="#cf9b3f"; GRAY="#586071"; INK="#1b1f27"

def box(ax,x,y,w,h,t,fc,ec,fs=9,bold=True,tc=INK):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.05",
                 lw=1.5,edgecolor=ec,facecolor=fc))
    ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=fs,fontweight="bold" if bold else "normal",color=tc)
def arr(ax,p1,p2,c=GRAY,lw=1.7,ls="-"):
    ax.add_patch(FancyArrowPatch(p1,p2,arrowstyle="-|>",mutation_scale=13,lw=lw,color=c,linestyle=ls,shrinkA=2,shrinkB=2))
def hlegend(ax,y,items):
    """Legend in reserved bottom space: 3 items in one row, 4 items in a 2x2 grid."""
    xs = [0.6,4.2,7.5] if len(items)<=3 else [0.9,5.6]
    perrow = len(xs)
    for i,(c,label) in enumerate(items):
        x=xs[i%perrow]; yy=y-(i//perrow)*0.52
        ax.add_patch(FancyBboxPatch((x,yy),0.28,0.24,boxstyle="round,pad=0.01",lw=0,facecolor=c))
        ax.text(x+0.42,yy+0.12,label,fontsize=8,va="center",color=GRAY)
LEG3=[(CLASS,"classical (deterministic)"),(LEARN,"learned (world model)"),(BLUE,"LLM (planning)")]
LEG4=LEG3+[(AMBER,"confidence gate")]

# ---- Fig 1: concept (the lab today + what we add) ---------------------------
def concept():
    fig,ax=plt.subplots(figsize=(7.4,4.2)); ax.set_xlim(0,10); ax.set_ylim(0.4,5.9); ax.axis("off")
    box(ax,0.3,4.9,9.4,0.7,"Copilot (LLM): turns a goal into an experiment plan and machine commands",
        "#e8eef7",BLUE,fs=8.5)
    for i,name in enumerate(["Synthesis","UPLC-MS","NMR"]):
        box(ax,0.5+i*3.2,0.8,2.6,1.15,name+"\n(instrument)","#e9f1ec",CLASS,fs=9.5)
    box(ax,3.4,2.85,3.2,1.15,"mobile arm\n+ onboard rack","#f6ece0",LEARN,fs=9.5,tc="#5a3210")
    arr(ax,(5.0,4.85),(5.0,4.05),c=BLUE)
    arr(ax,(4.4,2.8),(2.1,1.98),c=LEARN); arr(ax,(5.0,2.8),(5.0,1.98),c=LEARN); arr(ax,(5.8,2.8),(7.9,1.98),c=LEARN)
    ax.text(1.55,3.35,"carry and place\n(world model,\nthen visual servo)",ha="center",fontsize=7.6,color="#5a3210")
    fig.savefig(os.path.join(OUT,"fig_concept.png"),dpi=200,bbox_inches="tight"); plt.close(fig)

# ---- Fig 2: method architecture --------------------------------------------
def arch():
    fig,ax=plt.subplots(figsize=(7.4,5.0)); ax.set_xlim(0,10); ax.set_ylim(0.2,6.7); ax.axis("off")
    cx=5.0; W=8.6
    box(ax,cx-W/2,5.7,W,0.7,"Copilot LLM: decompose the goal into an action plan, re-plan on results","#e8eef7",BLUE,fs=8.5)
    box(ax,cx-W/2,4.55,W,0.7,"Deterministic scheduler: queue, resource locks, dispatch, idle or charge","#e9f1ec",CLASS,fs=8.5)
    box(ax,cx-W/2,3.25,W,0.8,"World-model arm control: variable coarse approach and place","#f6ece0",LEARN,fs=8.5,tc="#5a3210")
    gx,gy=cx,2.3
    ax.add_patch(Polygon([[gx,gy+0.55],[gx+1.95,gy],[gx,gy-0.55],[gx-1.95,gy]],closed=True,facecolor="#fbf1d8",edgecolor=AMBER,lw=1.6))
    ax.text(gx,gy,"energy or\nuncertainty gate",ha="center",va="center",fontsize=8.2,fontweight="bold",color="#6b4e12")
    box(ax,cx-W/2,0.6,W,0.8,"Classical control: known-geometry grasp, then visual-servo seat","#e9f1ec",CLASS,fs=8.5)
    for y1,y2 in [(5.7,5.3),(4.55,4.1),(3.25,2.9),(1.7,1.45)]: arr(ax,(cx,y1),(cx,y2))
    box(ax,0.7,gy-0.4,1.7,0.8,"retry or\nhuman","#fbf1d8",AMBER,fs=8,tc="#6b4e12")
    arr(ax,(gx-1.95,gy),(2.45,gy),c=AMBER); ax.text(2.78,gy+0.33,"unsure",ha="center",fontsize=7.5,color="#6b4e12")
    fig.savefig(os.path.join(OUT,"fig_arch.png"),dpi=200,bbox_inches="tight"); plt.close(fig)

# ---- Fig 3: precision hand-off ---------------------------------------------
def handoff():
    VIS="#2f7d8a"
    fig,ax=plt.subplots(figsize=(7.4,4.0)); ax.set_xlim(0,10); ax.set_ylim(-0.5,5.7); ax.axis("off")
    ax.text(5.0,5.35,"vision only precise seating",ha="center",fontsize=9.5,style="italic",color=VIS)
    box(ax,0.3,3.3,3.0,1.3,"world model\napproach and place\nabout 10 to 40 mm","#f6ece0",LEARN,fs=8.8,tc="#5a3210")
    box(ax,3.7,1.95,3.0,1.3,"wrist camera\nvisual servo\nabout 0.5 to 2 mm","#e4eff1",VIS,fs=8.8,tc="#1f4b54")
    box(ax,7.1,0.6,2.6,1.3,"seated via chamfer\nbelow about 0.5 mm","#e9f1ec",CLASS,fs=8.8)
    arr(ax,(3.3,3.45),(3.75,3.2),c=AMBER,lw=2.0); ax.text(4.7,3.95,"confidence gate",ha="left",fontsize=8,color="#6b4e12")
    arr(ax,(6.7,2.2),(7.15,1.9),c=CLASS,lw=2.0); ax.text(7.5,2.7,"chamfer (passive)",ha="left",fontsize=8,color=CLASS)
    ax.annotate("",xy=(9.8,0.05),xytext=(0.3,0.05),arrowprops=dict(arrowstyle="-|>",color=GRAY,lw=1.4))
    ax.text(5.0,-0.32,"end effector error decreasing",ha="center",fontsize=8,color=GRAY)
    fig.savefig(os.path.join(OUT,"fig_handoff.png"),dpi=200,bbox_inches="tight"); plt.close(fig)

if __name__=="__main__":
    concept(); arch(); handoff()
    for f in ("fig_concept.png","fig_arch.png","fig_handoff.png"):
        print(f, os.path.getsize(os.path.join(OUT,f)), "bytes")
