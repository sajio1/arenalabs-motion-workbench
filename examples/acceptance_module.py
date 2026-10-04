"""Interface example only. Never enabled automatically or labeled production QA."""
import json

def evaluate(result_manifest):
    result=json.loads(result_manifest.read_text(encoding='utf8'))
    # Replace with your approved FINAL-output acceptance implementation.
    return {'accepted':False,'reason':'Example module has no approved acceptance rules','pipeline':result['pipeline']}
