use rand::seq::IteratorRandom;
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

pub fn find_start_node(
    graph: &HashMap<String, HashSet<String>>,
    in_degree: &HashMap<String, usize>,
    out_degree: &HashMap<String, usize>,
) -> Option<String> {
    for node in out_degree.keys() {
        let out_deg = out_degree.get(node).unwrap_or(&0);
        let in_deg = in_degree.get(node).unwrap_or(&0);

        if *out_deg > *in_deg && (*out_deg - *in_deg) == 1 {
            return Some(node.clone());
        }
    }

    let mut rng = rand::thread_rng();
    graph.keys().choose(&mut rng).cloned()
}

pub fn find_eulerian_path(
    graph: &HashMap<String, HashSet<String>>,
    start_node: String,
) -> Vec<String> {
    let mut working_graph: HashMap<String, Vec<String>> = HashMap::new();
    for (node, neighbors) in graph {
        working_graph.insert(node.clone(), neighbors.iter().cloned().collect());
    }

    let mut stack: Vec<String> = vec![start_node];
    let mut path: Vec<String> = Vec::new();

    while let Some(current) = stack.last() {
        let has_neighbors = working_graph
            .get(current)
            .map(|neighbors| !neighbors.is_empty())
            .unwrap_or(false);

        if has_neighbors {
            let next_node = working_graph.get_mut(current).unwrap().pop().unwrap();
            stack.push(next_node);
        } else {
            path.push(stack.pop().unwrap());
        }
    }

    path.reverse();
    path
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

pub fn generate_reads(text: &str, read_length: usize, step: usize) -> Vec<String> {
    let mut reads = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    
    let mut i = 0;
    while i + read_length <= chars.len() {
        let read: String = chars[i..i + read_length].iter().collect();
        reads.push(read);
        i += step;
    }
    
    reads
}

