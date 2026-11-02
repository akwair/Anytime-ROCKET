"""Draw manuscript figures from the frozen experimental evidence."""
from pathlib import Path
import csv
import hashlib
import json
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon, FancyArrowPatch
from matplotlib.lines import Line2D

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
OUT = BASE/'figures'
OUT.mkdir(exist_ok=True)
E = json.loads((BASE/'evidence.json').read_text())
assert E == json.loads((BASE/'revisions/v1/evidence.json').read_text())
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 9, 'axes.labelsize': 9, 'xtick.labelsize': 8,
    'ytick.labelsize': 8.5, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': .65, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
    'text.color': '#29353C', 'axes.labelcolor': '#29353C',
    'xtick.color': '#465259', 'ytick.color': '#465259',
    'legend.frameon': False, 'savefig.facecolor': 'white',
})
C = dict(cyan='#C8E8F0', cream='#F8F0E7', rose='#D08888', purple='#9B80B0',
         taupe='#A8988F', orange='#E87928', blue='#257E9B', ink='#354850',
         lightpurple='#EDE6F3', lightrose='#F5E7E7', neutral='#A8B2B7')
COL = {'crc': C['blue'], 'ltt': C['orange']}
names = {'small_minirocket':'Small MiniRocket', 'ridge_prefix':'Natural prefix',
         'detach_minirocket':'SFD adaptation', 'pocket_minirocket':'POCKET adaptation',
         'strat_imp_static':'Stratified static', 'strat_imp_adaptive':'Anytime-ROCKET'}
qa = {'backend':'Python/matplotlib', 'n_datasets':113, 'seeds':list(range(5)),
      'source_summary_sha256':E['source_summary_sha256'], 'figures':[],
      'statistics':'Dataset means or equal-weight macro means; no intervals or new tests.',
      'integrity':'No generated observations, jitter, removed outliers or raster image manipulation.'}

def export(fig, stem):
    fig.canvas.draw()
    for fmt in ('pdf','svg','png'):
        fig.savefig(OUT/f'{stem}.{fmt}', dpi=600)
    fig.savefig(OUT/f'{stem}_preview.png', dpi=160)
    qa['figures'].append({'name':stem, 'width_inches':fig.get_figwidth(),
                          'height_inches':fig.get_figheight(), 'dpi':600})
    plt.close(fig)

def csvout(name, rows):
    with (OUT/name).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

def rect(ax, x,y,w,h,fc='white',ec='#BAC6CC',r=.012,lw=.75,**kw):
    p=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',
                    facecolor=fc,edgecolor=ec,linewidth=lw,**kw)
    ax.add_patch(p)
    return p

def text(ax,x,y,s,size=8,ha='center',**kw):
    return ax.text(x,y,s,fontsize=size,ha=ha,va='center',**kw)

def arrow(ax,x1,y1,x2,y2,color=None,style='-',lw=1,**kw):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=8,
                                color=color or C['ink'],linewidth=lw,linestyle=style,**kw))

def grid(ax,x,y,w,h,fc,cols=5,rows=4,depth=.009):
    ax.add_patch(Polygon([(x,y+h),(x+depth,y+h+depth),(x+w+depth,y+h+depth),(x+w,y+h)],
                         facecolor=fc,edgecolor='#88999F',lw=.45,alpha=.75))
    ax.add_patch(Polygon([(x+w,y),(x+w+depth,y+depth),(x+w+depth,y+h+depth),(x+w,y+h)],
                         facecolor=fc,edgecolor='#88999F',lw=.45,alpha=.9))
    ax.add_patch(Rectangle((x,y),w,h,facecolor=fc,edgecolor='#88999F',lw=.5))
    for i in range(1,cols): ax.plot([x+w*i/cols]*2,[y,y+h],color='#8DA4AC',lw=.3)
    for i in range(1,rows): ax.plot([x,x+w],[y+h*i/rows]*2,color='#8DA4AC',lw=.3)

def legend_handles():
    return [Line2D([],[],color=COL['crc'],marker='o',lw=0,label='CRC'),
            Line2D([],[],color=COL['ltt'],marker='s',markerfacecolor='white',lw=0,label='LTT')]

# Figure 1: schematic only. Feature blocks encode structure, not observed values.
fig=plt.figure(figsize=(7.2,5.0))
ax=fig.add_axes([0,0,1,1]); ax.set(xlim=(0,1),ylim=(0,1)); ax.axis('off')
rect(ax,.015,.700,.970,.283,C['cream'],ec='#E3D8CA',r=.017)
rect(ax,.015,.379,.970,.303,'#F1F8FA',ec='#CEE2E7',r=.017)
rect(ax,.015,.015,.970,.346,'#F4F0F8',ec='#DAD1E3',r=.017)
text(ax,.033,.953,'a',11,ha='left',weight='bold')
text(ax,.058,.953,'Computation units and dilation-stratified ordering',9.5,ha='left',weight='bold')
text(ax,.032,.650,'b',11,ha='left',weight='bold')
text(ax,.057,.650,'Fit the prefix classifiers and calibrate a common exit threshold',9.5,ha='left',weight='bold')
text(ax,.032,.330,'c',11,ha='left',weight='bold')
text(ax,.057,.330,'Evaluate cumulative prefixes, then exit or continue',9.5,ha='left',weight='bold')

# Unit structure.
rect(ax,.036,.737,.224,.177,'white',ec='#D9CCBD')
text(ax,.148,.892,'One kernel–dilation unit',8.3,weight='bold')
grid(ax,.055,.806,.070,.049,C['cyan'],cols=7,rows=2)
text(ax,.090,.775,'response',7)
arrow(ax,.138,.831,.173,.831)
for j,c in enumerate([C['rose'],C['purple'],C['taupe']]):
    rect(ax,.184,.792+j*.029,.049,.020,c,ec='none',r=.002)
text(ax,.209,.764,'PPV features',7)
text(ax,.365,.910,'Rank within each dilation',8.1,weight='bold')
for row,(lab,c) in enumerate([('$d_1$',C['cyan']),('$d_2$',C['rose']),('$d_3$',C['purple'])]):
    yy=.851-row*.047
    text(ax,.298,yy+.012,lab,8.5)
    for j in range(3):
        rect(ax,.328+j*.047,yy,.036,.026,c,ec='white',r=.003)
        text(ax,.346+j*.047,yy+.013,str(j+1),6.5,color=C['ink'])
    text(ax,.483,yy+.013,'…',9)
arrow(ax,.514,.823,.574,.823,color=C['orange'])
text(ax,.745,.910,'Round-robin unit sequence',8.1,weight='bold')
for j,c in enumerate([C['cyan'],C['rose'],C['purple']]*3):
    rect(ax,.592+j*.037,.799,.030,.052,c,ec='white',r=.003)
text(ax,.755,.764,'Higher-ranked units enter first at each scale',7.3)

# Fitting/calibration lanes, with input roles shown separately.
rect(ax,.038,.522,.170,.075,C['cream'],ec='#DFD1BF')
text(ax,.123,.560,'Labeled fit data\n$D_{\mathrm{fit}}$',8.4)
rect(ax,.038,.413,.170,.075,'white',ec='#BED9E2')
text(ax,.123,.451,'Unlabeled inputs\n$D_{\mathrm{cal}}$',8.4)
rect(ax,.272,.495,.221,.112,C['lightpurple'],ec='#CDBBDD')
text(ax,.382,.551,'MiniRocket + standardization\nFull head → unit order\nSeparate prefix heads',7.6)
arrow(ax,.209,.559,.269,.559)
arrow(ax,.211,.450,.307,.450,color=C['blue'])
text(ax,.395,.450,'Prefix and full predictions',7.5)
arrow(ax,.382,.493,.382,.468,color=C['purple'])
ax.plot([.487,.520],[.450,.450],color=C['blue'],lw=1)
grid(ax,.560,.481,.093,.108,C['cyan'],cols=6,rows=5)
text(ax,.608,.451,'Disagreement losses',7.2)
text(ax,.608,.421,'against the full head',7)
arrow(ax,.520,.450,.520,.532,color=C['blue'])
arrow(ax,.520,.532,.557,.532,color=C['blue'])
arrow(ax,.667,.532,.711,.532,color=C['blue'])
rect(ax,.720,.486,.226,.111,'white',ec='#B4C8CE')
text(ax,.833,.568,'LTT: fixed-sequence test',7.8)
text(ax,.833,.536,'CRC: monotone loss envelope',7.6)
text(ax,.833,.505,'Select threshold $\hat{\lambda}$',8.2,weight='bold',color=C['blue'])
text(ax,.302,.410,'Fit stage',7.1,color='#7D648E')
text(ax,.841,.414,'Independent calibration',7.2,color=C['blue'])

# Deployment. The bypass is visually distinct from finite-threshold continuation.
rect(ax,.039,.178,.106,.070,'white',ec='#C8B8D4')
text(ax,.092,.214,'Observed\nseries $x$',8)
arrow(ax,.145,.216,.210,.216)
centers=[.285,.564,.863]
for c,w,lab,head,shade in zip(centers,[.062,.110,.152],['Prefix 1','Prefix $j$','Full prefix'],
                             ['$h_1(x)$','$h_j(x)$','$f(x)$'],[C['cyan'],C['cyan'],C['purple']]):
    grid(ax,c-w/2,.202,w,.051,shade,cols=max(3,round(w*55)),rows=3)
    text(ax,c,.279,lab,8.1,weight='bold')
    arrow(ax,c,.198,c,.171,color='#7D648E')
    rect(ax,c-.055,.126,.110,.043,'white',ec='#BCA9CA',r=.007)
    text(ax,c,.148,head,8.5)
arrow(ax,.331,.226,.493,.226,color=C['purple'])
arrow(ax,.631,.226,.774,.226,color=C['purple'])
text(ax,.412,.246,'continue',7.1,color='#7D648E')
text(ax,.703,.246,'continue',7.1,color='#7D648E')
text(ax,.413,.207,'…',10)
text(ax,.704,.207,'…',10)
for c,j in zip(centers[:2],['1','j']):
    arrow(ax,c,.124,c,.093,color=C['orange'])
    text(ax,c,.073,f'Exit if $m_{j}(x)\\geq\\hat{{\\lambda}}$',7.7,color='#995C28')
    text(ax,c,.040,'Return this prediction',7.0)
arrow(ax,.863,.124,.863,.093,color=C['purple'])
text(ax,.863,.073,'Return full prediction',7.7)
ax.plot([.092,.092,.941,.941],[.251,.300,.300,.256],color='#9885A5',ls=(0,(3,2)),lw=.7)
text(ax,.710,.299,'$\hat{\lambda}=\infty$: direct full-model fallback',7.3,color='#6E5C7B',
     bbox={'facecolor':'#F4F0F8','edgecolor':'none','pad':1.7})
arrow(ax,.941,.256,.941,.250,color='#9885A5',lw=.7)
export(fig,'fig1_overview')

# Figure 2: macro performance of every method.
rows=[]
for m in names:
    for rule in ['crc','ltt']:
        r=E['tables'][f'{rule}_0.01'][m]
        rows.append(dict(method=m,rule=rule,target_percent=1,cost=r['cost'],
                         disagreement_percent=100*r['disagree'],delta_accuracy_pp=r['delta_acc_pp']))
csvout('fig2_source.csv',rows)
fig,axes=plt.subplots(1,2,figsize=(7.2,3.45),gridspec_kw={'width_ratios':[1,1]})
fig.subplots_adjust(left=.245,right=.975,bottom=.17,top=.83,wspace=.24)
ys=np.arange(6)[::-1]
for ax,metric,title,xlabel in zip(axes,['cost','disagreement_percent'],
                               ['a  Modeled relative cost','b  Observed disagreement'],
                               ['Relative cost','Disagreement (%)']):
    ax.axhspan(-.4,.4,color='#EFF7FA',zorder=0)
    for i,m in enumerate(names):
        vals=[next(r[metric] for r in rows if r['method']==m and r['rule']==q) for q in ['crc','ltt']]
        ax.plot(vals,[ys[i]]*2,color='#B9C3C8',lw=1.2,zorder=1)
        ax.scatter(vals[0],ys[i],c=COL['crc'],s=33,zorder=3,edgecolor='white',linewidth=.5)
        ax.scatter(vals[1],ys[i],facecolor='white',edgecolor=COL['ltt'],s=30,marker='s',zorder=3,lw=1.1)
    ax.set(yticks=ys,ylim=(-.6,5.6),xlim=(0,1.055),xlabel=xlabel)
    ax.set_title(title,loc='left',fontsize=9.2,fontweight='bold',pad=12)
    ax.axvline(1,color='#8E9B9F',ls=(0,(3,3)),lw=.8)
    ax.set_xticks(np.arange(0,1.01,.2))
    ax.grid(axis='x',color='#E6EAEC',lw=.6)
    ax.set_axisbelow(True); ax.tick_params(axis='y',length=0)
axes[0].set_yticklabels(list(names.values()))
axes[1].set_yticklabels([])
fig.legend(handles=legend_handles(),loc='upper center',ncol=2,bbox_to_anchor=(.61,1.015),fontsize=8.5)
export(fig,'fig2_main')

# Figure 3: independently reconstruct paired dataset means from all raw runs.
by=defaultdict(list)
hashes={}
for p in sorted((ROOT/'results/server/results/baselines').glob('*.json')):
    r=json.loads(p.read_text()); by[r['dataset']].append(r)
    hashes[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
assert len(by)==113 and all(sorted(r['seed'] for r in rs)==list(range(5)) for rs in by.values())
paired=[]
for dataset,rs in sorted(by.items()):
    paired.append({'dataset':dataset,**{m:float(np.mean([r['families'][m]['crc']['0.01']['cost'] for r in rs])) for m in names}})
for m in names:
    assert np.isclose(np.mean([r[m] for r in paired]), E['tables']['crc_0.01'][m]['cost'],atol=1e-12,rtol=0)
csvout('fig3_source.csv',paired)
fig,axes=plt.subplots(1,2,figsize=(7.2,3.65),sharex=True,sharey=True)
fig.subplots_adjust(left=.10,right=.98,bottom=.16,top=.88,wspace=.24)
for ax,m,panel in zip(axes,['detach_minirocket','strat_imp_static'],['a','b']):
    x=np.array([r[m] for r in paired]); y=np.array([r['strat_imp_adaptive'] for r in paired])
    wins=int(np.sum(y<x-1e-12)); ties=int(np.sum(np.abs(y-x)<=1e-12)); losses=113-wins-ties
    expected=E['paired']['crc_0.01'][m]
    assert (wins,ties,losses)==(expected['adaptive_wins'],expected['ties'],expected['adaptive_losses'])
    ax.fill_between([0,1.08],[0,0],[0,1.08],color='#F3F8FA',zorder=0)
    ax.plot([0,1.08],[0,1.08],ls=(0,(4,3)),c='#A7B3B9',lw=.9)
    ax.scatter(x,y,s=21,c=COL['crc'],edgecolor='white',lw=.35,alpha=.80)
    ax.set(xlim=(0,1.08),ylim=(0,1.08),aspect='equal',xlabel=names[m]+' cost')
    ax.set_title(panel+'  '+names[m],loc='left',fontsize=9.2,weight='bold',pad=12)
    ax.text(.05,.93,f'{wins} lower / {ties} equal / {losses} higher',transform=ax.transAxes,fontsize=8.0)
    ax.set_xticks(np.arange(0,1.01,.2)); ax.set_yticks(np.arange(0,1.01,.2))
axes[0].set_ylabel('Anytime-ROCKET cost')
export(fig,'fig3_paired')

# Figure 4: parameter sensitivity with the same rule encoding.
budget=[]; ordering=[]
for rule in ['crc','ltt']:
    for eps in ['0.005','0.01','0.02','0.05']:
        r=E['budget'][f'{rule}_{eps}']
        budget.append(dict(rule=rule,target_percent=float(eps)*100,cost=r['cost']))
    for order in ['importance','random','strat','strat_imp']:
        ordering.append(dict(rule=rule,order=order,cost=E['ordering'][f'{order}_{rule}'][0]))
csvout('fig4a_source.csv',budget); csvout('fig4b_source.csv',ordering)
fig=plt.figure(figsize=(7.2,3.45))
ax1=fig.add_axes([.10,.20,.33,.61]); ax2=fig.add_axes([.71,.20,.265,.61])
for rule,marker,ls,face in [('crc','o','-',COL['crc']),('ltt','s','--','white')]:
    r=[x for x in budget if x['rule']==rule]
    ax1.plot([x['target_percent'] for x in r],[x['cost'] for x in r],marker=marker,ls=ls,
             color=COL[rule],markerfacecolor=face,ms=5,lw=1.3)
ax1.set(xscale='log',xticks=[.5,1,2,5],xticklabels=['0.5','1','2','5'],
        xlabel='Disagreement target (%)',ylabel='Modeled relative cost',ylim=(0,1.0))
ax1.minorticks_off();ax1.grid(axis='y',color='#E4E9EB',lw=.6)
ax1.set_title('a  Risk-budget sensitivity',loc='left',fontsize=9.1,weight='bold',pad=12)
labels=['Importance','Random','Stratified random','Stratified importance']
for i,order in enumerate(['importance','random','strat','strat_imp']):
    vals=[E['ordering'][f'{order}_{q}'][0] for q in ['crc','ltt']]
    ax2.plot(vals,[3-i]*2,color='#BAC3C8',lw=1.2)
    ax2.scatter(vals[0],3-i,c=COL['crc'],s=33,zorder=3)
    ax2.scatter(vals[1],3-i,facecolor='white',edgecolor=COL['ltt'],s=29,marker='s',zorder=3)
ax2.axhspan(-.38,.38,color='#EFF7FA',zorder=0)
ax2.set(yticks=[3,2,1,0],yticklabels=labels,ylim=(-.6,3.6),xlim=(.50,.82),
        xlabel='Modeled relative cost',xticks=[.5,.6,.7,.8])
ax2.tick_params(axis='y',length=0,labelsize=7.8)
ax2.set_title('b  Ordering at a 1% target',loc='left',fontsize=9.1,weight='bold',pad=12)
ax2.grid(axis='x',color='#E4E9EB',lw=.6);ax2.set_axisbelow(True)
fig.legend(handles=legend_handles(),loc='upper center',ncol=2,bbox_to_anchor=(.51,1.005),fontsize=8.5)
export(fig,'fig4_sensitivity')

(OUT/'raw_input_hashes.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
(OUT/'figure_validation.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')
print('Created 4 figures in PDF, SVG and 600-dpi PNG, with 5 source-data CSV files.')
