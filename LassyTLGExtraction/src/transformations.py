from itertools import groupby, chain
from xml.etree.cElementTree import Element, ElementTree

from LassyTLGExtraction.src.graphutils import *


def sort_dags(dags: List[DAG]) -> List[DAG]:
    def dag_to_key(dag_: DAG) -> Any:
        leaves_ = order_nodes(dag_, set(filter(dag_.is_leaf, dag_.nodes)))
        return list(map(lambda leaf_: tuple(map(int, (dag_.attribs[leaf_]['begin'],
                                                      dag_.attribs[leaf_]['end'],
                                                      dag_.attribs[leaf_]['id']))),
                        leaves_))
    return sorted(dags, key=lambda dag: dag_to_key(dag))


def rename_dag_src(dags: List[DAG]) -> List[DAG]:
    metas = [None if dag.meta is None else {**dag.meta, **{'src': dag.meta['src'] + '_' + str(i)}}
             for i, dag in enumerate(dags)]

    return list(map(lambda dag, meta: DAG(nodes=dag.nodes,
                                          edges=dag.edges,
                                          attribs=dag.attribs,
                                          meta=meta),
                    dags,
                    metas))


def get_sentence(dag: DAG) -> List[str]:
    leaves = order_nodes(dag, set(filter(dag.is_leaf, dag.nodes)))
    return list(map(lambda leaf: dag.attribs[leaf]['word'], leaves))


def identify_nodes(nodes: Set[Element]) -> Dict[str, str]:
    coindexed = list(filter(lambda elem: 'index' in elem.attrib.keys(), nodes))
    all_mutual_indices = {i: [node for node in group] for i, group in
                          groupby(sorted(coindexed, key=lambda elem: elem.attrib['index']),
                                  key=lambda elem: elem.attrib['index'])}
    identifying_index = {i: fst(list(map(lambda elem: elem.attrib['id'],
                                         filter(lambda elem: 'cat' in elem.attrib.keys() or 'word' in
                                                             elem.attrib.keys(),
                                                elements))))
                         for i, elements in all_mutual_indices.items()}
    return {n.attrib['id']: identifying_index[n.attrib['index']] if 'index' in n.attrib.keys() else n.attrib['id']
            for n in nodes}


def convert_to_dag(tree: ElementTree, meta: Optional[Any] = None) -> DAG:
    nodes = set(tree.iter('node'))
    identifying_indices = identify_nodes(nodes)
    edges = [Edge(source.attrib['id'], target.attrib['id'], target.attrib['rel'])
             for source in nodes for target in source.findall('node')]
    edges = filter(lambda edge: edge.dep != '--', edges)
    edges = set(map(lambda edge: Edge(identifying_indices[edge.source], identifying_indices[edge.target], edge.dep),
                    edges))
    occurring_indices = set.union(set([edge.source for edge in edges]), set([edge.target for edge in edges]))
    occuring_nodes = filter(lambda node: node.attrib['id'] in occurring_indices or ('pt' in node.attrib.keys()
                                                                                    and node.attrib['pt'] != 'let'),
                            nodes)
    attribs = {node.attrib['id']: {k: v for k, v in node.attrib.items() if k != 'rel'} for node in occuring_nodes}
    return DAG(nodes=set(attribs.keys()), edges=edges, attribs=attribs, meta=meta)


def _cats_of_type(dag: DAG, cat: str, nodes: Optional[Nodes] = None) -> Nodes:
    if nodes is None:
        nodes = dag.nodes
    return set(filter(lambda node: 'cat' in dag.attribs[node] and dag.attribs[node]['cat'] == cat, nodes))


def order_nodes(dag: DAG, nodes: Nodes) -> List[Node]:
    return sorted(nodes, key=lambda node: tuple(map(int, (dag.attribs[node]['begin'],
                                                          dag.attribs[node]['end'],
                                                          dag.attribs[node]['id']))))


def majority_vote(dag: DAG, nodes: Nodes, pos_set: str = 'pt') -> str:
    def get_vote(node_: Node) -> str:
        if pos_set in dag.attribs[node_].keys():
            return dag.attribs[node_][pos_set]
        elif dag.attribs[node_]['cat'] == 'conj':
            return majority_vote(dag, dag.successors(node_), pos_set)
        else:
            return dag.attribs[node_]['cat']

    votes = list(map(lambda n: get_vote(n), nodes))
    votes = sorted(votes)
    votecounts = list(map(lambda v: (fst(v), len(list(snd(v)))), groupby(votes, key=lambda x: x)))
    votecounts = sorted(votecounts, key=lambda pair: snd(pair), reverse=True)
    votes = set(votes)
    if 'smain' in votes:
        return 'smain'
    elif 'np' in votes or 'n' in votes:
        return 'np'
    elif 'ap' in votes or 'a' in votes:
        return 'ap'
    else:
        return fst(fst(votecounts))


def remove_abstract_arguments(dag: DAG) -> DAG:
    candidates = {'su', 'obj', 'obj1', 'obj2', 'sup', 'pobj'}
    sentential_cats = {'sv1', 'smain', 'ssub', 'inf'}

    def has_sentential_parent(node: Node) -> bool:
        def is_sentential(node_: Node) -> True:
            if dag.attribs[node_]['cat'] in sentential_cats:
                return True
            elif dag.attribs[node_]['cat'] == 'conj':
                return any(list(map(is_sentential, dag.predecessors(node_))))
            return False

        return any(list(map(lambda pred: is_sentential(pred), dag.predecessors(node))))

    def is_candidate_dep(edge: Edge) -> bool:
        return edge.dep in candidates

    def is_coindexed(node: Node) -> bool:
        return len(dag.incoming(node)) > 1

    def is_inf_or_ppart(node: Node) -> bool:
        return dag.attribs[node]['cat'] in {'ppart', 'inf'}

    for_removal = set(filter(lambda e: is_candidate_dep(e)
                                       and is_coindexed(e.target)
                                       and is_inf_or_ppart(e.source)
                                       and has_sentential_parent(e.target),
                             dag.edges))

    return DAG(nodes=dag.nodes, edges=dag.edges.difference(for_removal), attribs=dag.attribs, meta=dag.meta)


def collapse_mwu(dag: DAG) -> DAG:
    dag = DAG(nodes=dag.nodes, edges=dag.edges, attribs=dag.attribs, meta=dag.meta)
    mwus = _cats_of_type(dag, 'mwu')
    successors = list(map(lambda mwu: order_nodes(dag, dag.successors(mwu)), mwus))
    collapsed_texts = list(map(lambda suc: ' '.join([dag.attribs[s]['word'] for s in suc]), successors))
    for mwu, succ, text in zip(mwus, successors, collapsed_texts):
        dag.attribs[mwu]['word'] = text
        del dag.attribs[mwu]['cat']
        dag.attribs[mwu]['pt'] = majority_vote(dag, set(succ), 'pt')
        dag.attribs[mwu]['pos'] = majority_vote(dag, set(succ), 'pos')
    to_delete = set(list(chain.from_iterable(map(dag.outgoing, mwus))))
    return dag.remove_edges(lambda e: e not in to_delete)


def refine_body(dag: DAG) -> DAG:
    bodies = list(dag.get_edges('body'))
    to_add, to_remove = set(), set()
    for body in bodies:
        common_source = dag.outgoing(body.source)
        match = list(filter(lambda edge: edge.dep in ('cmp', 'rhd', 'whd'), common_source))
        if len(match) != 1:
            from src.viz import ToGraphViz
            import pdb
            ToGraphViz()(dag)
            pdb.set_trace()
        match = fst(match)
        new_dep = match.dep + '_body'
        to_add.add(Edge(source=body.source, target=body.target, dep=new_dep))
        to_remove.add(body)
    return DAG(nodes=dag.nodes, edges=dag.edges.difference(to_remove).union(to_add), attribs=dag.attribs, meta=dag.meta)


def remove_secondary_dets(dag: DAG) -> DAG:
    def tw_case(target: Node) -> bool:
        return True if dag.attribs[target]['pt'] == 'tw' or dag.attribs[target]['word'] == 'beide' else False
    to_add, to_remove = set(), set()

    detgroups = list(dag.get_edges('det'))
    detgroups = groupby(sorted(detgroups, key=lambda edge: edge.source), key=lambda edge: edge.source)
    detgroups = list(map(lambda group: list(snd(group)), detgroups))
    detgroups = list(filter(lambda group: len(group) > 1, detgroups))
    for detgroup in detgroups:
        targets = list(map(lambda edge: edge.target, detgroup))
        if all(list(map(dag.is_leaf, targets))):
            tw = filter(lambda x: tw_case(snd(x)), zip(detgroup, targets))
            tw = set(map(fst, tw))
            to_remove = to_remove.union(tw)
            to_add = to_add.union(set(map(lambda edge: Edge(source=edge.source, target=edge.target, dep='mod'), tw)))
        else:
            detp = filter(lambda x: not dag.is_leaf(snd(x)), zip(detgroup, targets))
            detp = set(map(fst, detp))
            to_remove = to_remove.union(detp)
            to_add = to_add.union(set(map(lambda edge: Edge(source=edge.source, target=edge.target, dep='mod'),
                                          detp)))
    return DAG(nodes=dag.nodes, edges=dag.edges.difference(to_remove).union(to_add), attribs=dag.attribs, meta=dag.meta)


def swap_dp_headedness(dag: DAG) -> DAG:
    dag = remove_secondary_dets(dag)
    to_add, to_remove = set(), set()

    dets = list(dag.get_edges('det'))
    matches = list(map(lambda edge: fst(list(filter(lambda out: out.dep == 'hd', dag.outgoing(edge.source)))), dets))

    for d, m in zip(dets, matches):
        to_remove.add(d)
        to_remove.add(m)
        to_add.add(Edge(source=d.source, target=d.target, dep='hd'))
        to_add.add(Edge(source=m.source, target=m.target, dep='invdet'))

    return DAG(nodes=dag.nodes, edges=dag.edges.difference(to_remove).union(to_add), attribs=dag.attribs, meta=dag.meta)


def reattatch_conj_mods(dag: DAG, mod_candidates: Iterable[Dep] = ('mod', 'app', 'predm')) -> DAG:
    to_add, to_remove = set(), set()

    modgroups = sorted(set.union(*list(map(lambda m: set(dag.get_edges(m)), mod_candidates))),
                       key=lambda edge: edge.target)
    modgroups = list(map(lambda g: list(snd(g)), groupby(modgroups, key=lambda edge: edge.target)))
    modgroups = list(filter(lambda g: len(g) > 1, modgroups))
    for modgroup in modgroups:
        sources = list(map(lambda edge: edge.source, modgroup))
        common_ancestor = dag.first_common_predecessor(sources)
        if common_ancestor is None:
            raise ValueError('No common ancestor.')
        to_remove = to_remove.union(set(modgroup))
        to_add.add(Edge(source=common_ancestor, target=fst(modgroup).target, dep=fst(modgroup).dep))
    return DAG(nodes=dag.nodes, edges=dag.edges.difference(to_remove).union(to_add), attribs=dag.attribs, meta=dag.meta)


def remove_headless_branches(dag: DAG, cats_to_remove: Iterable[str] = ('du',),
                             deps_to_remove: Iterable[str] = ('dp', 'sat', 'nucl', 'tag')) -> List[DAG]:
    bad_nodes = set.union(*(list(map(lambda cat: _cats_of_type(dag, cat), cats_to_remove))))
    dag = dag.remove_nodes(lambda n: n not in bad_nodes)
    bad_edges = set.union(*list(map(lambda dep: set(dag.get_edges(dep)), deps_to_remove)))
    dag = dag.remove_edges(lambda edge: edge not in bad_edges, normalize=False)
    dags = dag.get_rooted_subgraphs()
    dags = list(map(remove_non_leaves, dags))
    return list(filter(good_sample, dags))


def remove_non_leaves(dag: DAG) -> DAG:
    def non_leaf(node_: Node) -> bool:
        return not len(dag.outgoing(node_)) and 'cat' in dag.attribs[node_].keys()
    return dag.remove_nodes(lambda node: not non_leaf(node))


def good_sample(dag: DAG) -> bool:
    if len(dag.nodes) == 1:
        node = fst(list(dag.nodes))
        if 'cat' in dag.attribs[node].keys():
            return False
        elif dag.attribs[node]['pt'] == 'vg':
            return False
        else:
            return True
    else:
        coords = list(map(lambda edge: dag.points_to(edge.source).difference({edge.target}), dag.get_edges('crd')))
        coords = list(map(lambda nodes: list(filter(dag.is_leaf, nodes)), coords))
        coords = list(filter(lambda leaves: len(leaves) < 2, coords))
        if len(coords):
            return False

        return True


class Transformation(object):
    def __init__(self):
        self.cats_to_remove = {'du'}
        self.deps_to_remove = {'dp', 'sat', 'nucl', 'tag'}
        self.mod_deps = {'mod', 'app', 'predm'}

    def __call__(self, tree: ElementTree, meta: Optional[Any] = None) -> List[DAG]:
        dag = convert_to_dag(tree, meta)
        dag = collapse_mwu(dag)
        dags = remove_headless_branches(dag, self.cats_to_remove, self.deps_to_remove)
        dags = list(map(remove_abstract_arguments, dags))
        dags = list(map(refine_body, dags))
        dags = list(map(swap_dp_headedness, dags))
        dags = list(map(lambda dag: reattatch_conj_mods(dag, self.mod_deps), dags))
        dags = list(map(lambda dag: dag.remove_oneways(), dags))
        dags = list(filter(lambda dag: not dag.is_empty(), dags))
        dags = list(map(lambda dag: dag.remove_oneways(),
                        list(chain.from_iterable(map(lambda dag: dag.get_subgraphs(), dags)))))
        return rename_dag_src(sort_dags(dags))


def test(samples=100):
    from src.lassy import Lassy
    L = Lassy()
    T = Transformation()

    meta = [{'src': last(L[i][1].split('/'))} for i in range(samples)]

    return list(chain.from_iterable(list(map(T, list(map(lambda i: L[i][2], range(samples))), meta))))
