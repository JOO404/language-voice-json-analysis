"""Measure supplied voice samples using the same librosa analysis as interviews."""
import argparse
from pathlib import Path

from pipeline.interview_analysis.analyze import main as analyze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cleaned', type=Path, help='Optional separate folder of aligned script JSON')
    args = parser.parse_args()
    options = ['--project', str(args.project), '--out', str(args.out),
               '--stages', 'prosody', '--timing-mode', 'positive']
    if args.cleaned:
        options.extend(['--cleaned', str(args.cleaned)])
    return analyze(options)
