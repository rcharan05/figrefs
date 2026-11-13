"""Figure recall: GROBID 0.9.0 alone, GROBID + caption recovery, pdfparse.
Found = same number and caption token-F1 >= 0.4, or >= 70% of the predicted caption's words (>= 5) inside the true caption
(accepts GROBID's title-only captions). usage: python eval/figure_recall.py arxiv|biorxiv"""
import sys, os, re, json, collections
R='/home/charan/proj/figrefs'
sys.path[:0]=[R, R+'/eval', R+'/eval/figsum']
from sentence_gt import latex_sentences, jats_sentences
from data import load_article_tei, toks, tf1
from sentence_eval import caption_alignment
from figrefs.pdfparse import parse_pdf
import pymupdf
which=sys.argv[1]
root = R+'/eval/arxiv' if which=='arxiv' else '/home/charan/bench/biorxiv'
tei_name = 'grobid_090_sent.tei.xml' if which=='arxiv' else 'grobid_090.tei.xml'
def ok(gt, pred):
    g, p = toks(gt), toks(re.sub(r'^(Fig\.?|Figure)\s*\d+\s*[.:|]?', '', pred))
    return tf1(g, p) >= 0.4 or (len(p) >= 5 and len(g & p) / len(p) >= 0.7)
tot=collections.Counter(); per=[]
for d in sorted(os.listdir(root)):
    p=os.path.join(root,d); pdf=p+'/paper.pdf'; tei=p+'/'+tei_name
    if not (os.path.exists(pdf) and os.path.exists(tei)): continue
    try:
        gt = latex_sentences(p+'/src') if which=='arxiv' else jats_sentences(p+'/paper.xml')
    except Exception: continue
    if not gt or len(gt['figures'])<2: continue
    doc=pymupdf.open(pdf)
    if doc.page_count>60: continue
    if which=='arxiv' and caption_alignment(gt, doc) < 0.6: continue
    G={f['number']:f['caption'] for f in gt['figures']}
    referenced={m['figure'] for m in gt['mentions']}
    a0=load_article_tei(d, tei, None)            # GROBID only
    a1=load_article_tei(d, tei, pdf)             # + caption recovery
    try:
        pp={int(f.id):f.caption for f in parse_pdf(pdf).figures if f.kind=='figure' and f.id.isdigit()}
    except Exception: pp={}
    row={'paper':d,'gt':len(G),'referenced':len(referenced)}
    for name,F in (('grobid',a0.figures),('grobid+recovery',a1.figures),('pdfparse',pp)):
        hit={n for n,c in G.items() if n in F and ok(c, F[n])}
        row[name]=len(hit); row[name+'_ref']=len(hit&referenced)
        row[name+'_zero']=int(len(hit)==0)
        tot[name]+=len(hit); tot[name+'_ref']+=len(hit&referenced); tot[name+'_zero']+=int(len(hit)==0)
    # union: grobid+recovery, else pdfparse
    U=dict(pp); U.update({n:c for n,c in a1.figures.items() if c})
    hit={n for n,c in G.items() if n in U and ok(c, U[n])}
    tot['union']+=len(hit); tot['union_ref']+=len(hit&referenced); tot['union_zero']+=int(len(hit)==0)
    tot['gt']+=len(G); tot['ref']+=len(referenced); tot['papers']+=1
    per.append(row)
print(which, dict(tot))
for k in ('grobid','grobid+recovery','pdfparse','union'):
    print(f"  {k:16s} figure recall {tot[k]/tot['gt']:.3f}  (referenced figures {tot[k+'_ref']/tot['ref']:.3f})  papers with zero figures {tot[k+'_zero']}/{tot['papers']}")
json.dump(per, open(f'/home/charan/proj/figrefs/eval/figrec_{which}.json','w'), indent=1)
