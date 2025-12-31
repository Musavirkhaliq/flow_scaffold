#!/usr/bin/env python3
"""
Analyze sequence quality to detect repetitive patterns and other issues.

This script checks for:
- Homopolymeric runs (repetitive amino acids)
- Amino acid composition
- Sequence complexity (Shannon entropy)
- Low-complexity regions
"""
import sys
from pathlib import Path
from collections import Counter
import numpy as np
import argparse


def parse_fasta(fasta_file):
    """Parse FASTA file and return sequences"""
    sequences = []
    current_seq = []
    current_header = None
    
    with open(fasta_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_seq:
                    sequences.append({
                        'header': current_header,
                        'sequence': ''.join(current_seq)
                    })
                current_header = line[1:]
                current_seq = []
            else:
                current_seq.append(line)
        
        if current_seq:
            sequences.append({
                'header': current_header,
                'sequence': ''.join(current_seq)
            })
    
    return sequences


def find_homopolymeric_runs(sequence, min_length=4):
    """Find runs of identical amino acids"""
    runs = []
    i = 0
    while i < len(sequence):
        aa = sequence[i]
        run_length = 1
        j = i + 1
        while j < len(sequence) and sequence[j] == aa:
            run_length += 1
            j += 1
        
        if run_length >= min_length:
            runs.append({
                'aa': aa,
                'length': run_length,
                'start': i,
                'end': j
            })
        
        i = j if run_length > 1 else i + 1
    
    return runs


def calculate_shannon_entropy(sequence):
    """Calculate Shannon entropy of sequence"""
    counts = Counter(sequence)
    total = len(sequence)
    entropy = 0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * np.log2(p)
    return entropy


def analyze_composition(sequence):
    """Analyze amino acid composition"""
    counts = Counter(sequence)
    total = len(sequence)
    composition = {aa: count/total * 100 for aa, count in counts.items()}
    return composition


def detect_low_complexity(sequence, window=20, threshold=2.0):
    """Detect low-complexity regions using sliding window entropy"""
    low_complexity_regions = []
    
    for i in range(len(sequence) - window + 1):
        window_seq = sequence[i:i+window]
        entropy = calculate_shannon_entropy(window_seq)
        
        if entropy < threshold:
            low_complexity_regions.append({
                'start': i,
                'end': i + window,
                'entropy': entropy,
                'sequence': window_seq
            })
    
    return low_complexity_regions


def analyze_sequence(seq_dict):
    """Comprehensive sequence analysis"""
    sequence = seq_dict['sequence']
    header = seq_dict['header']
    
    # Basic stats
    length = len(sequence)
    
    # Homopolymeric runs
    runs = find_homopolymeric_runs(sequence, min_length=4)
    
    # Entropy
    entropy = calculate_shannon_entropy(sequence)
    
    # Composition
    composition = analyze_composition(sequence)
    
    # Low complexity
    low_complexity = detect_low_complexity(sequence)
    
    # Quality flags
    flags = []
    if len(runs) > 0:
        flags.append(f"⚠️  {len(runs)} homopolymeric runs")
    if entropy < 2.5:
        flags.append(f"⚠️  Low entropy ({entropy:.2f})")
    if len(low_complexity) > length * 0.2:
        flags.append(f"⚠️  High low-complexity ({len(low_complexity)/length*100:.1f}%)")
    
    # Check for over-represented amino acids
    for aa, pct in composition.items():
        if pct > 25:
            flags.append(f"⚠️  {aa} over-represented ({pct:.1f}%)")
    
    return {
        'header': header,
        'length': length,
        'entropy': entropy,
        'composition': composition,
        'runs': runs,
        'low_complexity': low_complexity,
        'flags': flags,
        'quality': 'GOOD' if len(flags) == 0 else 'POOR'
    }


def print_analysis(analysis, verbose=False):
    """Print analysis results"""
    print(f"\n{'='*80}")
    print(f"Sequence: {analysis['header'][:60]}")
    print(f"{'='*80}")
    print(f"Length: {analysis['length']} aa")
    print(f"Entropy: {analysis['entropy']:.2f} bits (>2.5 is good)")
    print(f"Quality: {analysis['quality']}")
    
    if analysis['flags']:
        print(f"\nIssues:")
        for flag in analysis['flags']:
            print(f"  {flag}")
    else:
        print(f"\n✅ No issues detected")
    
    if verbose:
        print(f"\nComposition:")
        sorted_comp = sorted(analysis['composition'].items(), key=lambda x: x[1], reverse=True)
        for aa, pct in sorted_comp[:10]:
            print(f"  {aa}: {pct:5.1f}%")
        
        if analysis['runs']:
            print(f"\nHomopolymeric runs:")
            for run in analysis['runs'][:5]:
                print(f"  {run['aa']} × {run['length']} at position {run['start']}")
        
        if analysis['low_complexity']:
            print(f"\nLow-complexity regions: {len(analysis['low_complexity'])}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze sequence quality and detect repetitive patterns"
    )
    parser.add_argument(
        "fasta_file",
        type=str,
        help="FASTA file to analyze"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print summary statistics only"
    )
    
    args = parser.parse_args()
    
    # Parse FASTA
    sequences = parse_fasta(args.fasta_file)
    
    if not sequences:
        print(f"No sequences found in {args.fasta_file}")
        return
    
    print(f"\n{'='*80}")
    print(f"SEQUENCE QUALITY ANALYSIS")
    print(f"{'='*80}")
    print(f"File: {args.fasta_file}")
    print(f"Sequences: {len(sequences)}")
    
    # Analyze all sequences
    analyses = [analyze_sequence(seq) for seq in sequences]
    
    # Summary statistics
    good_count = sum(1 for a in analyses if a['quality'] == 'GOOD')
    poor_count = len(analyses) - good_count
    avg_entropy = np.mean([a['entropy'] for a in analyses])
    avg_runs = np.mean([len(a['runs']) for a in analyses])
    
    print(f"\nSummary:")
    print(f"  Good quality: {good_count}/{len(analyses)} ({good_count/len(analyses)*100:.1f}%)")
    print(f"  Poor quality: {poor_count}/{len(analyses)} ({poor_count/len(analyses)*100:.1f}%)")
    print(f"  Average entropy: {avg_entropy:.2f} bits")
    print(f"  Average homopolymeric runs: {avg_runs:.1f}")
    
    if not args.summary:
        # Print individual analyses
        for analysis in analyses:
            print_analysis(analysis, verbose=args.verbose)
    
    # Overall assessment
    print(f"\n{'='*80}")
    print(f"OVERALL ASSESSMENT")
    print(f"{'='*80}")
    
    if poor_count / len(analyses) > 0.5:
        print("❌ POOR: >50% of sequences have quality issues")
        print("\nRecommendations:")
        print("  1. Increase ProteinMPNN temperature (try 0.3-0.5)")
        print("  2. Check backbone quality (Ramachandran plots)")
        print("  3. Retrain model with more epochs (100+)")
        print("  4. Increase sampling steps (100+)")
    elif poor_count / len(analyses) > 0.2:
        print("⚠️  MODERATE: 20-50% of sequences have quality issues")
        print("\nRecommendations:")
        print("  1. Consider increasing ProteinMPNN temperature")
        print("  2. Validate backbone quality")
    else:
        print("✅ GOOD: <20% of sequences have quality issues")
        print("\nSequences appear to be of good quality!")
    
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
