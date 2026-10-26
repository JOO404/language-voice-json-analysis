import copy
import json
from pathlib import Path
import sys
import unittest

PACKAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE))
from pipeline.transcript_review.validate_json import check
from pipeline.interview_analysis.bar_profile import fixed_scores
from pipeline.interview_analysis.analyze import prepare_inputs


class InputChecks(unittest.TestCase):
    def setUp(self):
        self.data=json.loads((PACKAGE/'example/project/data/cleaned/demo_voice/sample.json').read_text(encoding='utf-8'))

    def test_invalid_timestamps_are_detected(self):
        for start,end in [(float('nan'),1),(-1,1),(1,0),(0,float('inf'))]:
            data=copy.deepcopy(self.data)
            data['words'][0].update(start=start,end=end)
            self.assertTrue(check(data)[0])

    def test_explicitly_excluded_bad_interval_is_a_warning(self):
        data=copy.deepcopy(self.data)
        data['words'][0].update(start=1,end=0,acoustic_eligible=False)
        errors,warnings=check(data)
        self.assertFalse(errors)
        self.assertTrue(warnings)

    def test_eligible_mode_cannot_silently_assume_review(self):
        with self.assertRaisesRegex(ValueError,'acoustic_eligible'):
            prepare_inputs(PACKAGE/'example/project','eligible')

    def test_bar_reference_is_frozen_and_clipped(self):
        reference=json.loads((PACKAGE/'config/bar_reference.json').read_text(encoding='utf-8'))
        means={k:v['mean'] for k,v in reference['dimensions'].items()}
        self.assertTrue(all(v==3. for v in fixed_scores(means,reference).values()))
        high={k:v['mean']+10*v['sample_std'] for k,v in reference['dimensions'].items()}
        self.assertTrue(all(v==5. for v in fixed_scores(high,reference).values()))


if __name__=='__main__':unittest.main(verbosity=2)
