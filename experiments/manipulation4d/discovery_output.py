"""Normalize declared VLM JSON formats without inventing object categories."""

POLICY='qwen-visible-objects-v3-canonical'
EXCLUDED={'wall','floor','ceiling','window','door','sink','faucet','table','chair','sofa','couch','bed','cabinet','countertop','television','tv','mirror','shelf','joystick','controller','game controller','vr controller','robot gripper','robot arm','hand','arm','person'}
ALIASES={'remote':'remote control','chopping board':'cutting board','cell phone':'phone','mobile phone':'phone','drinking glass':'glass'}
ALIASES.update({'coffee cup':'cup','drinking cup':'cup','cup lid':'lid','toy block':'building block','lego block':'building block','lego brick':'building block','paint brush':'paintbrush','dish drying rack':'dish rack','drinking glasses':'glass'})
ALIASES.update({'rubber ducky':'rubber duck','rubber duckie':'rubber duck','cellphone':'phone','smartphone':'phone','tv remote':'remote control'})
# Deliberately bounded vocabulary: never strip arbitrary adjectives or stem words
# (e.g. glass -> glas). Functional subtypes such as measuring cup stay distinct.
PLURALS={x+'s':x for x in ['cup','lid','bottle','plate','bowl','container','box','tool','utensil','spoon','fork','book','phone','remote control','paper cup','paper plate','building block','block','hand','arm','robot gripper','robot arm','controller']}
PLURALS.update({x+'s':x for x in ['brick','tray','card','coin','ring','paintbrush','pencil','pen','sponge','cloth','towel','plant','curtain','laptop','tablet','basket','spatula','controller','game controller','vr controller','camera']})
PLURALS.update({'boxes':'box','knives':'knife','dominoes':'domino','brushes':'brush','dishes':'dish','batteries':'battery'})
PLURALS.update({x+'s':x for x in ['duck','ball','balloon','toy','bracelet','hoodie','blender','pan','pot','rack','dispenser','bag','pouch','packet','package','bowl','mug','jar','can','screw','bolt','nut','washer','hammer','screwdriver','cable','cord','rope','shoe','sock','shirt','jacket','apple','orange','banana','onion','carrot','egg','sandwich','cookie','button','bottle cap','cap','key','flower','vase','device','comb','broom','brush']})
PLURALS.update({'duckies':'ducky'})
MATERIALS={'paper','plastic','ceramic','porcelain','glass','metal','wooden','cardboard','disposable','stainless steel'}
MATERIAL_CLASSES={'cup','plate','bowl','bottle','lid','container','box','spoon','fork','knife'}
COLORS={'white','black','red','blue','green','yellow','orange','purple','pink','brown','gray','grey','clear','transparent'}
OBJECT_CLASSES=MATERIAL_CLASSES|{'tray','card','coin','ring','paintbrush','pencil','pen','sponge','cloth','towel','book','building block','block','basket','spatula','phone','laptop','tablet'}

def canonical_label(label):
 text=' '.join(label.lower().replace('_',' ').split())
 text=ALIASES.get(text,text)
 # Singularize only a recognized terminal noun, preserving modifiers.
 words=text.split()
 if words and words[-1] in PLURALS:words[-1]=PLURALS[words[-1]]
 text=' '.join(words)
 text=ALIASES.get(text,text)
 # Appearance attributes are metadata, not additional segmentation passes.
 while text.split(' ',1)[0] in COLORS and ' ' in text:
  remainder=text.split(' ',1)[1]
  if canonical_label(remainder) not in OBJECT_CLASSES|EXCLUDED:break
  text=remainder
 for material in sorted(MATERIALS,key=len,reverse=True):
  if text.startswith(material+' ') and text[len(material)+1:] in MATERIAL_CLASSES:
   return text[len(material)+1:]
 return text

def valid_label(label):
 text=label.strip().lower()
 return bool(text) and len(text)<=60 and not any(x in text for x in ['short concrete','visual category','object name','category name','one sentence','string','unknown','unidentified','not visible'])
def normalize_discovery(value):
 rows=value if isinstance(value,list) else [value]
 if not rows or any(not isinstance(r,dict) for r in rows):raise ValueError('Invalid discovery JSON')
 objects=[]
 for row in rows:
  declared=row.get('objects',[])
  if not isinstance(declared,list):raise ValueError('Invalid object proposal list')
  for item in declared:
   label=item if isinstance(item,str) else item.get('category') if isinstance(item,dict) else None
   if not isinstance(label,str):raise ValueError('Invalid object category')
   objects.append(label)
  if row.get('category'):objects.append(row['category'])
 if any(not isinstance(x,str) or not valid_label(x) for x in objects):raise ValueError('Invalid or placeholder object category')
 aliases={}
 for label in objects:
  canonical=canonical_label(label)
  if canonical not in EXCLUDED and label not in aliases.setdefault(canonical,[]):aliases[canonical].append(label)
 result=dict(rows[0]);result['objects']=list(aliases)[:12]
 result['object_aliases']={k:aliases[k] for k in result['objects']}
 if not result.get('task'):result['task']=result.get('instruction','')
 return result
