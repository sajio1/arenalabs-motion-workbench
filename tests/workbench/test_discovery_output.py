from arenalab_workbench.discovery_output import normalize_discovery


def test_discovery_output():
    assert normalize_discovery({'objects':['Cup','cup','lid'],'task':'close a cup'})['objects']==['cup','lid']
    assert normalize_discovery({'objects':[],'task':'walk through an empty room'})['objects']==[]
    assert normalize_discovery([{'category':'book','instruction':'turn a page'}])['objects']==['book']
    for value in [{'objects':['short concrete singular visual category']},{'objects':['object name']},{'objects':'cup'},{'objects':[None]}]:
     try:normalize_discovery(value)
     except ValueError:pass
     else:raise AssertionError(value)
    print('PASS: actual names accepted; empty evidence retained; placeholder/malformed output rejected')
