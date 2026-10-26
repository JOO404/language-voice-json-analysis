"""Run the research pipeline by task: review, interview, or voices."""
import argparse
import importlib
import sys


WORKFLOWS = {
    'review': ('Audio/transcript preparation and BAD/overlap review', {
        'prepare': ('pipeline.transcript_review.prepare_audio', 'Extract WAV from clips or supplied intervals'),
        'transcribe': ('pipeline.transcript_review.transcribe', 'Transcribe Korean audio or check a supplied script'),
        'align': ('pipeline.transcript_review.align', 'Align words to audio timestamps'),
        'format': ('pipeline.transcript_review.format_json', 'Format already compatible word JSON'),
        'validate': ('pipeline.transcript_review.validate_json', 'Check JSON fields and word intervals'),
        'listen': ('pipeline.transcript_review.review', 'Replay audio/video and mark BAD or overlap (Windows)'),
        'timing': ('pipeline.transcript_review.timing_qc', 'Screen reviewed word timings for acoustic analysis'),
    }),
    'interview': ('Interview language, acoustic and participant summaries', {
        'analyze': ('pipeline.interview_analysis.analyze', 'Measure prosody and classify text emotion/intent'),
        'profile': ('pipeline.interview_analysis.bar_profile', 'Aggregate participant BAR indicators'),
    }),
    'voices': ('Supplied-script voice sample analysis and comparison', {
        'analyze': ('pipeline.voice_samples.analyze', 'Measure voice acoustics with the shared analysis method'),
        'compare': ('pipeline.voice_samples.compare', 'Summarize clips, voices, script checks and timing sensitivity'),
    }),
}


def main(argv=None):
    # Keep Korean text and QC symbols readable in redirected Windows terminals.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    groups = parser.add_subparsers(dest='task', metavar='{review,interview,voices}')
    parsers = {}
    for task, (description, actions) in WORKFLOWS.items():
        group = groups.add_parser(task, help=description, description=description)
        parsers[task] = group
        subcommands = group.add_subparsers(dest='action')
        for action, (module, help_text) in actions.items():
            command = subcommands.add_parser(action, help=help_text, add_help=False)
            command.set_defaults(module=module)
    args, remaining = parser.parse_known_args(argv)
    if not getattr(args, 'module', None):
        if remaining:
            parser.error('unrecognized arguments: ' + ' '.join(remaining))
        parsers.get(args.task, parser).print_help()
        return 0
    old_argv = sys.argv
    try:
        sys.argv = [f'{parser.prog} {args.task} {args.action}', *remaining]
        return importlib.import_module(args.module).main()
    finally:
        sys.argv = old_argv


if __name__ == '__main__':
    raise SystemExit(main())
