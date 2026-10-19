"""Add common demo navigation without modifying Isaac playback or recordings."""
from pathlib import Path
import argparse, re
parser=argparse.ArgumentParser();parser.add_argument('path',type=Path);args=parser.parse_args()
s=args.path.read_text()
nav='<nav class="arenalab-demo-nav" aria-label="Demos"><a href="/locomotion">Locomotion</a><a href="/manipulation">Manipulation</a><a href="/operator/" aria-current="page">Operator</a></nav>'
header='<header class="arena-header"><a class="arena-brand" href="/locomotion"><img src="/web/arenalab-logo-transparent.png" alt="">ArenaLabs</a>'+nav+'<span class="arena-context">OpenArm</span></header>'
s=re.sub(r'<title>.*?</title>','<title>ArenaLabs · Operator</title>',s,count=1,flags=re.S)
if '<base ' not in s:s=s.replace('</title>','</title><base href="/isaac/">',1)
s=re.sub(r'<link rel="stylesheet" href="/web/demo-nav.css[^"]*">','',s)
s=s.replace('</title>','</title><link rel="stylesheet" href="/web/demo-nav.css?v=20261015-3">',1)
s,count=re.subn(r'<header(?:\s[^>]*)?>.*?</header>',lambda _:header,s,count=1,flags=re.S);assert count==1
if '<body' not in s:s=s.replace('<header class="arena-header">','<body class="operator-page"><header class="arena-header">',1)
elif 'operator-page' not in s:s=s.replace('<body','<body class="operator-page"',1)
args.path.write_text(s)
