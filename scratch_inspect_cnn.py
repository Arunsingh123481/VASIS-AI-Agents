import json

atoms = json.load(open(r'e:\Vasis AI\.pageindex_cache\2f898aaa197e\atoms.json', encoding='utf-8'))
tail = [a for a in atoms if a.get('page_number', a.get('page_num', 0)) >= 9]
with open('output.txt', 'w', encoding='utf-8') as f:
    for a in sorted(tail, key=lambda x: int(x['atom_id'])):
        pg = a.get('page_number', a.get('page_num', 0))
        f.write(f"atom_id={a['atom_id']} page={pg} text={repr(a['text'][:150])}\n")
