"""Validate the manuscript artifact against its preserved first draft."""
from pathlib import Path
import hashlib
import json
import re
from pypdf import PdfReader
from PIL import Image, ImageDraw

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]
old=(BASE/'revisions/v1/main.tex').read_text(encoding='utf-8')
new=(BASE/'main.tex').read_text(encoding='utf-8')
equations=lambda s:re.findall(r'\\begin\{equation\}(.*?)\\end\{equation\}',s,re.S)
tabulars=lambda s:re.findall(r'\\begin\{tabular\}(.*?)\\end\{tabular\}',s,re.S)
citations=lambda s:sorted(set(k for g in re.findall(r'\\cite\{([^}]+)\}',s) for k in g.split(',')))
assert equations(old)==equations(new)
assert tabulars(old)==tabulars(new)
assert citations(old)==citations(new)
assert json.loads((BASE/'evidence.json').read_text())==json.loads((BASE/'revisions/v1/evidence.json').read_text())
assert new.count(r'\begin{figure}')==4
assert new.count(r'\begin{table}')==6
assert '@@' not in new
assert new.count('{')==new.count('}')
for f in re.findall(r'\\includegraphics\[[^]]*\]\{([^}]+)\}',new):
    assert (BASE/f).is_file(),f
log=(BASE/'main.log').read_text(encoding='utf-8',errors='replace')
assert not re.search(r'Overfull|undefined|Rerun to get|LaTeX Error',log)
pdf=PdfReader(BASE/'main.pdf')
texts=[p.extract_text() for p in pdf.pages]
assert all(len(t)>500 for t in texts), 'Near-empty page'
assert not any('\ufffd' in t or '??' in t for t in texts)
assert len(pdf.pages)==16
abstract=re.search(r'\\abstract\{(.*?)\}\s*\\keywords',new,re.S).group(1)
bibkeys=set(re.findall(r'@\w+\{([^,]+),',(BASE/'references.bib').read_text()))
assert bibkeys==set(citations(new))
figure_pages={}
for i,t in enumerate(texts):
    for n in re.findall(r'Fig\.\s*(\d)\s+(?:Anytime-ROCKET workflow|Comparison at|Dataset-level modeled|Risk-budget)',t):
        figure_pages[n]=i+1
data=dict(pages=len(pdf.pages),abstract_words=len(abstract.split()),keywords=6,tables=6,figures=4,
          references=len(citations(new)),equations_identical_to_v1=True,
          table_cells_identical_to_v1=True,evidence_identical_to_v1=True,
          cited_reference_set_identical_to_v1=True,missing_citations=[],unused_references=[],
          overfull_boxes=False,unresolved_references=False,replacement_glyphs=0,
          draft_placeholders=new.count(r'\todo{'),figure_pages=figure_pages,
          pdf_sha256=hashlib.sha256((BASE/'main.pdf').read_bytes()).hexdigest(),
          visual_review='Pending final rendered-page inspection')
(BASE/'validation.json').write_text(json.dumps(data,indent=2),encoding='utf-8')

folder=ROOT/'tmp/pdfs/paa_revised_review'
pages=sorted(folder.glob('final-*.png'))
assert len(pages)==len(pdf.pages)
for start in range(0,len(pages),6):
    sheet=Image.new('RGB',(1380,1280),'#E4E7EA'); draw=ImageDraw.Draw(sheet)
    for j,p in enumerate(pages[start:start+6]):
        im=Image.open(p).convert('RGB');im.thumbnail((440,600))
        x=j%3*460+10;y=j//3*640+30
        sheet.paste(im,(x,y));draw.text((x,y-20),f'Page {start+j+1}',fill='black')
    sheet.save(folder/f'final_contact_{start//6+1}.png')
print(json.dumps(data,indent=2))
