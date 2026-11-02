"""Check editorial revisions against the frozen start of the review cycle."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import re

BASE = Path(__file__).resolve().parent
START = BASE/'adversarial/pre_review'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def extracted(pattern, value):
    return re.findall(pattern, value, flags=re.DOTALL)

def validate(stage='pdf'):
    old = (START/'main.tex').read_text(encoding='utf-8')
    current = (BASE/'main.tex').read_text(encoding='utf-8')
    checks = {}
    checks['equations_unchanged'] = extracted(r'\\begin\{equation\}(.*?)\\end\{equation\}', old) == extracted(r'\\begin\{equation\}(.*?)\\end\{equation\}', current)
    checks['table_cells_unchanged'] = extracted(r'\\begin\{tabular\}(.*?)\\end\{tabular\}', old) == extracted(r'\\begin\{tabular\}(.*?)\\end\{tabular\}', current)
    cited = lambda s: sorted(set(k for g in extracted(r'\\cite\{([^}]+)\}',s) for k in g.split(',')))
    checks['citation_set_unchanged'] = cited(old) == cited(current)
    checks['bibliography_unchanged'] = sha(START/'references.bib') == sha(BASE/'references.bib')
    def generator_logic(path):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'table':
                if node.args and isinstance(node.args[0], ast.Constant):
                    node.args[0].value = '<caption text>'
        return ast.dump(tree,include_attributes=False)
    checks['table_generator_logic_unchanged'] = generator_logic(START/'build_evidence.py') == generator_logic(BASE/'build_evidence.py')
    asset_hashes = json.loads((START/'figure_hashes.json').read_text(encoding='utf-8'))
    checks['figure_assets_unchanged'] = all(sha(BASE/name)==expected for name,expected in asset_hashes.items())
    checks['evidence_unchanged'] = json.loads((START/'evidence.json').read_text()) == json.loads((BASE/'evidence.json').read_text())
    checks['four_figures_six_tables'] = current.count(r'\begin{figure}') == 4 and current.count(r'\begin{table}') == 6
    checks['table_markers_resolved'] = '@@' not in current
    checks['balanced_braces'] = current.count('{') == current.count('}')
    checks['figure_files_exist'] = all((BASE/p).is_file() for p in extracted(r'\\includegraphics\[[^]]*\]\{([^}]+)\}',current))
    # Bibliography and author fact placeholders are deliberately outside the editorial work.
    old_declarations = old.split(r'\section*{Statements and Declarations}',1)[1].split(r'\appendix',1)[0]
    new_declarations = current.split(r'\section*{Statements and Declarations}',1)[1].split(r'\appendix',1)[0]
    checks['author_declarations_untouched'] = old_declarations == new_declarations
    result = {'stage':stage,'checks':checks,
              'draft_sha256':sha(BASE/'draft.tex.in'),
              'build_evidence_sha256':sha(BASE/'build_evidence.py'),
              'main_tex_sha256':sha(BASE/'main.tex'),
              'equations':len(extracted(r'\\begin\{equation\}',current)),
              'tables':6,'figures':4,'references':len(cited(current))}
    if stage == 'pdf':
        from pypdf import PdfReader
        log = (BASE/'main.log').read_text(encoding='utf-8',errors='replace')
        checks['no_overfull_boxes'] = 'Overfull' not in log
        checks['no_unresolved_references'] = not bool(re.search(r'undefined|Rerun to get|LaTeX Error|Fatal error',log))
        pdf = PdfReader(BASE/'main.pdf')
        texts = [p.extract_text() or '' for p in pdf.pages]
        checks['no_replacement_glyphs'] = not any('\ufffd' in t or '??' in t for t in texts)
        # A short last page is flagged for human review, rather than silently removed.
        result['short_pages_to_inspect'] = [i+1 for i,t in enumerate(texts) if len(t)<500]
        result['pages'] = len(pdf.pages)
        result['pdf_sha256'] = sha(BASE/'main.pdf')
        result['visual_review'] = 'Pending rendered-page inspection'
    result['passed'] = all(checks.values())
    (BASE/'adversarial'/f'{stage}_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    if not result['passed']:
        raise SystemExit('Validation failed: '+', '.join(k for k,v in checks.items() if not v))
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['source','pdf'],default='pdf')
    validate(parser.parse_args().stage)
