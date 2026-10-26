"""Compare the generated tone measurements with the recorded synthetic-fixture result."""
import argparse
from pathlib import Path
from analyze import read


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result',type=Path,required=True)
    args=parser.parse_args()
    package=Path(__file__).resolve().parents[1]
    expected=read(package/'example/expected_prosody.json')
    actual=read(args.result/'data/prosody/demo_voice/sample.json')
    for key in ['file_summary','words']:
        if actual.get(key)!=expected[key]:
            raise SystemExit('MISMATCH: '+key)
    print('PASS: all saved word measurements and clip summary values match exactly.')


if __name__=='__main__':main()
