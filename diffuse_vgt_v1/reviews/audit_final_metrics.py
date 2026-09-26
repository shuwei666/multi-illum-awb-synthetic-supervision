"""Independent CPU audit. No production imports, GT reads or GPU use."""
import csv, hashlib, json, math
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]; P=R.parent; D=R/'results'
hashes={}; checks=0; maxima={}
def sha(p):
    p=Path(p); h=hashlib.sha256(p.read_bytes()).hexdigest()
    hashes[str(p.relative_to(P))]=h
    return h
def read(p): return json.loads(Path(p).read_text())
def rows(p):
    with open(p,newline='') as f:return list(csv.DictReader(f))
def check(v,msg):
    global checks
    checks+=1
    assert v,msg
def near(a,b,t=1e-12,label='precise'):
    e=abs(float(a)-float(b)); maxima[label]=max(e,maxima.get(label,0))
    check(math.isfinite(e) and e<=t,f'{label}: {a} versus {b}')
def mean(x):return math.fsum(x)/len(x)
selection=read(R/'development_selection.json'); frozen=read(R/'frozen/manifest.json')
manifest=read(D/'evaluation_manifest.json'); start=read(D/'evaluation_start.json'); summary=read(D/'summary.json')
approval=read(R/'frozen/evaluation_approval.json'); prior=read(R/'reviews/completed_cohort_review.json')
b=manifest['bindings']; check(manifest['status']=='complete','completion')
check(start['bindings']==b,'start bindings')
for k,v in b.items():check(approval[k]==v and prior[k]==v and summary[k]==v,'binding '+k)
for f,key in [('frozen/manifest.json','freeze_manifest_sha256'),('development_selection.json','development_selection_sha256')]:check(sha(R/f)==b[key],key)
check(sha(R/'frozen/evaluation_approval.json')==manifest['approval_sha256']==start['approval_sha256'],'approval hash')
check(sha(R/'reviews/completed_cohort_review.json')==manifest['review_sha256']==start['review_sha256'],'review hash')
check(sha(D/'evaluation_manifest.json')==summary['evaluation_manifest_sha256'],'result manifest hash')
check(prior['artifact_verdict']=='PASS' and prior['contract_verdict']=='ADEQUATE','prior approval')
for f,h in b['implementation_hashes'].items():check(sha(P/f)==h,'implementation '+f)
for f,h in b['code_hashes'].items():
    q=R/f if f in ('train_arms.py','diffuse_synthesis.py','render_diffuse.py','cache_priors.py') else P/'code'/f
    if f=='domislovic_v2.py':q=P/'code/frozen'/f
    if f=='launch_round3.py':q=P/'orchestration'/f
    check(sha(q)==h,'code '+f)
for arm,files in b['completion_evidence_sha256'].items():
    for f,h in files.items():check(sha(R/'runs'/f'{arm}_seed0'/f)==h,'completion '+arm+f)
for arm,h in b['checkpoint_sha256'].items():
    q=P/'evidence/baseline_nikon_seed0_ckpt_final.pt' if arm=='baseline' else P/'round3/runs/O_seed0/ckpt_final.pt' if arm=='O' else R/'runs'/f'{arm}_seed0/ckpt_final.pt'
    check(sha(q)==h,'checkpoint '+arm)
reference=read(P/'round2/final_eval/evaluation_manifest.json')
check(sha(P/'round2/final_eval/evaluation_manifest.json')==b['official_reference_manifest_sha256'],'official reference')
sources=read(R/'frozen/sources.json');check(sha(R/'frozen/sources.json')==frozen['copies']['sources.json'],'sources hash')
check(manifest['live_metadata_sha256']=={k:sources[k] for k in ('split_sha256','meta_sha256')},'metadata binding')
check(selection==manifest['selection'] and selection['tie_order']==['O','A','B','C'],'selection frozen')
scores={}
for m in selection['tie_order']:
    a=selection['results'][m]['means'];ref=frozen['reference_means']
    scores[m]=.5*a['real_single']/ref['real_single']+.125*math.fsum(a[k]/ref[k] for k in ('global','sigmoid','blob','lowfreq'))
    near(scores[m],selection['scores'][m]);check(selection['checkpoint_sha256'][m]==b['checkpoint_sha256'][m],'dev checkpoint')
winner=min(selection['tie_order'],key=lambda m:scores[m]);check(winner==selection['winner']==summary['frozen_winner'],'winner')
check(selection['selected_at']<prior['reviewed_at']<start['created']<=manifest['completed'],'chronology')
expected={f'{s}_{m}{f}' for s in ('val','test') for m in ('baseline','O','A','B','C') for f in ('_summary.json','_per_image.csv','_patches.npz')}
check(set(manifest['evidence_sha256'])==expected and len(expected)==30,'inventory')
for f,h in manifest['evidence_sha256'].items():check(sha(D/f)==h,'evidence '+f)
for f,h in summary['paired_artifacts_sha256'].items():check(sha(D/f)==h,'paired hash')
metrics={m:{} for m in ('baseline','O','A','B','C')}; data={}; support={}; scenes={}; history={}
for s,n in [('val',394),('test',204)]:
    ids=manifest[s+'_image_ids'];check(ids==reference[s+'_image_ids'] and len(ids)==len(set(ids))==n,'official IDs')
    scenes[s]={i.rsplit('_',1)[0] for i in ids};check(not scenes[s]&set(sources['source_ids']),'train leakage')
    base_support=None
    for m in metrics:
        rr=rows(D/f'{s}_{m}_per_image.csv');a=np.load(D/f'{s}_{m}_patches.npz',allow_pickle=False)
        e=a['patch_angular_error'];v=a['valid_patch'];check(a['ids'].tolist()==ids==[r['id'] for r in rr],'CSV NPZ IDs')
        check(e.shape==v.shape==(n,256) and v.dtype==bool,'patch shape')
        check(np.isfinite(e[v]).all() and (e[v]>=0).all(),'finite error')
        sig=[(r['n_lights'],r['valid_patches'],r['valid_pixels']) for r in rr]
        if base_support is None:base_support=(v.copy(),sig)
        else:check(np.array_equal(v,base_support[0]) and sig==base_support[1],'model support')
        im=[]; sums=[]; counts=[]
        for i,r in enumerate(rr):
            check(r['model']==m and r['split']==s,'row identity')
            nl=int(r['n_lights']);check(nl in (1,2,3) and nl==len(r['id'].rsplit('_',1)[1]),'lights')
            cnt=int(v[i].sum());check(cnt==int(r['valid_patches']) and cnt>0 and int(r['valid_pixels'])>0,'counts')
            su=math.fsum(map(float,e[i,v[i]])); sums.append(su);counts.append(cnt);im.append(su/cnt)
            near(im[-1],r['patch_mean'],2e-5,'serialized_patch')
            check(math.isfinite(float(r['pixel_mean'])) and float(r['pixel_mean'])>=0,'pixel finite')
        single=read(D/f'{s}_{m}_summary.json');check(single==manifest['splits'][s][m],'single summary binding')
        check(single['checkpoint_sha256']==b['checkpoint_sha256'][m],'summary checkpoint')
        near(mean([float(r['pixel_mean']) for r in rr]),single['pixel_image_balanced_mean'])
        pc=sum(int(r['valid_pixels']) for r in rr);check(pc==single['pixel_count'],'pixel count')
        near(math.fsum(float(r['pixel_mean'])*int(r['valid_pixels']) for r in rr)/pc,single['pixel_pooled_mean'])
        near(mean(im),single['patch_image_balanced_mean'],2e-5,'serialized_patch')
        for g in ('all','single','multi'):
            ix=[i for i,r in enumerate(rr) if g=='all' or (int(r['n_lights'])==1)==(g=='single')];check(bool(ix),'nonempty')
            vals={'patch_pooled':math.fsum(sums[i] for i in ix)/sum(counts[i] for i in ix),'patch_image_balanced':mean([im[i] for i in ix]),'pixel_image_balanced':mean([float(rr[i]['pixel_mean']) for i in ix])}
            sk='patch_pooled' if g=='all' else 'patch_'+g
            near(vals['patch_pooled'],single[sk]['mean'],2e-5,'serialized_patch');check(single[sk]['count']==sum(counts[i] for i in ix),'patch count summary')
            support[s+'/'+g]={'images':len(ix),'patches':sum(counts[i] for i in ix),'pixels':sum(int(rr[i]['valid_pixels']) for i in ix)}
            for k,x in vals.items():metrics[m][s+'/'+g+'/'+k]=x;near(x,summary['metrics'][m][s+'/'+g+'/'+k])
        data[s,m]=rr
        if m in ('baseline','O'):
            old_npz=np.load(P/'round3/final_eval'/f'{s}_{m}_patches.npz',allow_pickle=False)
            old_rows=rows(P/'round3/final_eval'/f'{s}_{m}_per_image.csv')
            check(old_npz['ids'].tolist()==ids and np.array_equal(old_npz['valid_patch'],v),'historical support')
            check([(r['id'],r['n_lights'],r['valid_patches'],r['valid_pixels']) for r in old_rows]==[(r['id'],r['n_lights'],r['valid_patches'],r['valid_pixels']) for r in rr],'historical CSV support')
            for g in ('all','single','multi'):
                ix=[i for i,r in enumerate(old_rows) if g=='all' or (int(r['n_lights'])==1)==(g=='single')]
                old_im=[mean(list(map(float,old_npz['patch_angular_error'][i,old_npz['valid_patch'][i]]))) for i in ix]
                old_pool=math.fsum(float(x) for i in ix for x in old_npz['patch_angular_error'][i,old_npz['valid_patch'][i]])/sum(int(old_npz['valid_patch'][i].sum()) for i in ix)
                for k,x in [('patch_pooled',old_pool),('patch_image_balanced',mean(old_im)),('pixel_image_balanced',mean([float(old_rows[i]['pixel_mean']) for i in ix]))]:
                    near(metrics[m][s+'/'+g+'/'+k],x,2e-5,'historical_18_cells')
            if m=='baseline':
                official_npz=np.load(P/'round2/final_eval'/f'{s}_baseline_patches.npz',allow_pickle=False)
                check(official_npz['ids'].tolist()==ids and np.array_equal(official_npz['valid_patch'],v),'official valid positions')
            old=read(P/'round3/final_eval'/f'{s}_{m}_summary.json');delta={}
            for k in ('patch_pooled','patch_single','patch_multi'):
                delta[k]=single[k]['mean']-old[k]['mean'];near(single[k]['mean'],old[k]['mean'],2e-5,'historical')
            for k in ('patch_image_balanced_mean','pixel_image_balanced_mean','pixel_pooled_mean'):
                delta[k]=single[k]-old[k];near(single[k],old[k],2e-5,'historical')
            history[s+'/'+m]=delta;sha(P/'round3/final_eval'/f'{s}_{m}_summary.json')
check(not scenes['val']&scenes['test'],'val test leakage')
comparisons={}; contrasts={}
for m in ('O','A','B','C'):
    ds={k:metrics[m][k]-metrics['baseline'][k] for k in metrics[m]};passed=sum(x<0 for x in ds.values())
    check(passed==summary['comparisons'][m]['passed_cells'],'strict pass count')
    check((passed==18)==summary['comparisons'][m]['all_means_strictly_lower'],'strict flag')
    for k,x in ds.items():near(x,summary['comparisons'][m]['delta_degrees'][k])
    comparisons[m]={'passed_cells':passed,'delta_degrees':ds}
for c in ('A','C'):
    ds={k:metrics['B'][k]-metrics[c][k] for k in metrics[c]};contrasts['B-'+c]=ds
    for k,x in ds.items():near(x,summary['mechanism_delta_degrees']['B-'+c][k])
check(summary['outcome']==('SEED0_GATE_MET' if winner in ('A','B','C') and comparisons[winner]['passed_cells']==18 else 'FINAL_TARGET_NOT_MET'),'outcome')
check(summary['winner_passes_seed0_gate']==False and summary['multi_seed_goal_complete']==False,'performance flags')
paired={}
for s in ('val','test'):
    for c in ('A','C'):
        for b0,c0 in zip(data[s,'B'],data[s,c]):
            paired[s,'B-'+c,b0['id']]={k:float(b0[k])-float(c0[k]) for k in ('patch_mean','pixel_mean')}
pr=rows(D/'per_image_deltas.csv');check(len(pr)==len(paired) and len({(r['split'],r['contrast'],r['id']) for r in pr})==len(pr),'paired image inventory')
for r in pr:
    check(r['scene']==r['id'].rsplit('_',1)[0] and int(r['n_lights'])==len(r['id'].rsplit('_',1)[1]),'paired identity')
    for k,x in paired[r['split'],r['contrast'],r['id']].items():near(x,r[k+'_delta'])
sr=rows(D/'per_scene_deltas.csv');expected_scene={(s,c,sc,g) for s in scenes for c in ('B-A','B-C') for sc in scenes[s] for g in ('all','single','multi')}
check({(r['split'],r['contrast'],r['scene'],r['group']) for r in sr}==expected_scene and len(sr)==len(expected_scene),'scene inventory')
for r in sr:
    use=[x for x in pr if all(x[k]==r[k] for k in ('split','contrast','scene')) and (r['group']=='all' or (int(x['n_lights'])==1)==(r['group']=='single'))]
    check(len(use)==int(r['compositions']),'scene compositions')
    for k in ('patch_mean_delta','pixel_mean_delta'):near(mean([float(x[k]) for x in use]),r[k])
scene_means={}
for s in scenes:
    for c in ('B-A','B-C'):
        for g in ('all','single','multi'):
            use=[r for r in sr if (r['split'],r['contrast'],r['group'])==(s,c,g)]
            scene_means['/'.join((s,c,g))]={k:mean([float(x[k]) for x in use]) for k in ('patch_mean_delta','pixel_mean_delta')}
sha(Path(__file__));sha(D/'summary.json');sha(D/'evaluation_start.json')
result={'artifact_verdict':'PASS','contract_verdict':'ADEQUATE','scope':'saved final metric arithmetic, support and frozen bindings','performance_outcome':summary['outcome'],'maturity':'S1 traceable numerical audit; no S2 performance success','checks':checks,'tolerances':{'precise_degrees':1e-12,'serialized_patch_and_history_degrees':2e-5},'max_discrepancies':maxima,'metrics':metrics,'support':support,'scene_counts':{s:len(x) for s,x in scenes.items()},'development_scores':scores,'winner':winner,'selection_before_scoring_seconds':start['created']-selection['selected_at'],'comparisons':comparisons,'mechanism_deltas':contrasts,'scene_equal_weight_deltas':scene_means,'historical_summary_deltas':history,'paired_image_rows':len(pr),'paired_scene_rows':len(sr),'input_sha256':hashes,'isolation':'limited: same model and shared filesystem; fresh context; source read-only by convention','pixel_boundary':'CSV pixel means reaggregated; no raw GT native pixel error reconstruction','limitations':['Historical test exposure disclosed; chronology is recorded evidence, not proof of no prior access.','No significance or training seed uncertainty inference from seed0.','No new physical validation; TIFF lineage is user-accepted engineering assumption.']}
out=R/'reviews/final_metrics_review.json';out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:result[k] for k in ('artifact_verdict','contract_verdict','performance_outcome','checks','max_discrepancies','scene_counts','paired_image_rows','paired_scene_rows')},indent=2))
