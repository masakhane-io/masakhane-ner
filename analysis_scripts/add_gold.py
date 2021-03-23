import sys

with open(sys.argv[1], 'r') as goldin, open(sys.argv[2], 'r') as hypin:
  for linenum, (gold, hyp) in enumerate(zip(goldin, hypin)):
    gold, hyp = gold.strip(), hyp.strip()
    if not gold:
      if hyp: raise ValueError(f'misaligned "{gold}" and "{hyp}" on line {linenum}')
      print()
    else:
      goldword, goldtag = gold.split()
      hypword, hyptag = hyp.split()
      assert(goldword == hypword)
      print(f'{goldword} {goldtag} {hyptag}')
    
