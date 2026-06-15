use std::collections::{HashMap, HashSet};

pub fn build_de_bruijn_graph(reads: &[String], k: usize) -> HashMap<String, HashSet<String>> {
    let mut graph: HashMap<String, HashSet<String>> = HashMap::new();
    let total_reads = reads.len();
    
    for (i, read) in reads.iter().enumerate() {
        
        if i % 1000 == 0 && i > 0 {
            println!("  -> Przetwarzanie grafu: {} / {} odczytów ({:.1}%)",
                i, total_reads, (i as f64 / total_reads as f64) * 100.0);
        }
        let chars: Vec<char> = read.chars().collect();
        
        if chars.len() < k {
            continue;
        }

        for window in chars.windows(k) {
            let prefix: String = window[..k - 1].iter().collect();
            let suffix: String = window[1..].iter().collect();

            graph.entry(prefix).or_insert_with(HashSet::new).insert(suffix);
        }
    }

    graph
}

pub fn get_degrees(
    graph: &HashMap<String, HashSet<String>>,
) -> (HashMap<String, usize>, HashMap<String, usize>) {
    let mut in_degree: HashMap<String, usize> = HashMap::new();
    let mut out_degree: HashMap<String, usize> = HashMap::new();

    for (node, neighbors) in graph {
        *out_degree.entry(node.clone()).or_insert(0) += neighbors.len();

        for neighbor in neighbors {
            *in_degree.entry(neighbor.clone()).or_insert(0) += 1;
        }
    }

    (in_degree, out_degree)
}

fn is_1_in_1_out(
    node: &str,
    in_degree: &HashMap<String, usize>,
    out_degree: &HashMap<String, usize>,
) -> bool {
    in_degree.get(node).copied().unwrap_or(0) == 1
        && out_degree.get(node).copied().unwrap_or(0) == 1
}

fn sorted_neighbors(graph: &HashMap<String, HashSet<String>>, node: &str) -> Vec<String> {
    let mut neighbors: Vec<String> = graph
        .get(node)
        .map(|items| items.iter().cloned().collect())
        .unwrap_or_default();
    neighbors.sort();
    neighbors
}

fn extend_unitig(
    working_graph: &mut HashMap<String, Vec<String>>,
    in_degree: &HashMap<String, usize>,
    out_degree: &HashMap<String, usize>,
    start: String,
    next: String,
    stop_at_start: bool,
) -> Vec<String> {
    let mut path: Vec<String> = vec![start.clone(), next.clone()];
    let cycle_start = start;
    let mut current = next;

    loop {
        if stop_at_start && current == cycle_start {
            break;
        }

        if !is_1_in_1_out(&current, in_degree, out_degree) {
            break;
        }

        let next_node = if let Some(neighbors) = working_graph.get_mut(&current) {
            neighbors.pop()
        } else {
            None
        };

        if let Some(next_node) = next_node {
            path.push(next_node.clone());
            current = next_node;
        } else {
            break;
        }
    }

    path
}

pub fn extract_contig_paths(graph: &HashMap<String, HashSet<String>>) -> Vec<Vec<String>> {
    let (in_degree, out_degree) = get_degrees(graph);
    let mut contigs: Vec<Vec<String>> = Vec::new();

    let mut working_graph: HashMap<String, Vec<String>> = HashMap::new();
    for (node, neighbors) in graph {
        let mut sorted_neighbors: Vec<String> = neighbors.iter().cloned().collect();
        sorted_neighbors.sort_by(|a, b| b.cmp(a)); 
        working_graph.insert(node.clone(), sorted_neighbors);
    }

    let mut start_nodes: Vec<String> = graph.keys()
        .filter(|node| {
            out_degree.get(*node).copied().unwrap_or(0) > 0
                && !is_1_in_1_out(node, &in_degree, &out_degree)
        })
        .cloned()
        .collect();
    start_nodes.sort();

    for start in start_nodes {
        while let Some(next) = working_graph.get_mut(&start).and_then(|n| n.pop()) {
            let contig = extend_unitig(
                &mut working_graph,
                &in_degree,
                &out_degree,
                start.clone(),
                next,
                false,
            );
            contigs.push(contig);
        }
    }

    let mut all_nodes: Vec<String> = graph.keys().cloned().collect();
    all_nodes.sort();

    for start in all_nodes {
        while let Some(next) = working_graph.get_mut(&start).and_then(|n| n.pop()) {
            let contig = extend_unitig(
                &mut working_graph,
                &in_degree,
                &out_degree,
                start.clone(),
                next,
                true,
            );
            contigs.push(contig);
        }
    }

    contigs
}

pub fn assemble(path: &[String]) -> String {
    if path.is_empty() {
        return String::new();
    }

    let mut result = path[0].clone();

    for node in path.iter().skip(1) {
        let last_char = node.chars().last().unwrap();
        result.push(last_char);
    }

    result
}

pub fn assemble_contigs(paths: &[Vec<String>]) -> Vec<String> {
    paths.iter().map(|path| assemble(path)).collect()
}
