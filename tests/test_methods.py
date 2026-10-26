import unittest,tempfile,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code/_core'))
from timing import inspect_word,validate_override
from intent_rules import detect_intent

class IntentTests(unittest.TestCase):
    def test_documented_expressions(self):
        cases=[('동의합니다.','affirmation'),('저도 동의해요.','affirmation'),('동의하지 않습니다.','negation'),('동의할 수 없어요.','negation'),('안 동의해요.','negation'),('동의 못 합니다.','negation'),('동의하시나요?','question'),('동의합니다?','question'),('동의라는 단어를 배웠어요.','statement'),('동의한다고 말할 것 같아요.','statement'),('동의한다는 뜻은 아니에요.','negation'),('보통 쳐주는 것 같아요 맞아요.','affirmation'),('맞아요.','affirmation'),('그렇죠.','affirmation'),('저도 그렇죠.','affirmation'),('어디에서 동의하시나요?','question'),('무슨 일을 했어요.','statement'),('안 할게요.','negation'),('못 해요.','negation'),('음.','hesitation'),('글쎄요.','hesitation'),('안녕하세요.','statement'),('저는 운동해요.','statement'),('','unknown')]
        for text,expected in cases:
            with self.subTest(text=text):self.assertEqual(detect_intent(text)[0],expected)
    def test_prosody_cannot_reverse_text(self):
        for slope in [-100,100]:
            p={'prosody_reliable':True,'words':[{'f0_slope':slope}]}
            for text,expected in [('맞아요.','affirmation'),('동의하지 않습니다.','negation'),('평범한 문장이에요.','statement')]:
                self.assertEqual(detect_intent(text,p)[0],expected)
    def test_narrative_agreement_and_bare_question(self):
        for text,expected in [('친구들도 만나고 그래요.','statement'),('바로 뭐라고 그래요.','statement'),('그래요.','affirmation'),('왜 이렇게 늦었어','question'),('왜 왔어','question'),('왜 그런지 설명했어.','statement')]:
            self.assertEqual(detect_intent(text)[0],expected,text)

class TimingTests(unittest.TestCase):
    clip={'duration':10,'speech_intervals':[[1,2],[5,6]]}
    def word(self,a,b):return {'type':'word','text':'가','start':a,'end':b}
    def test_operational_thresholds(self):
        self.assertIn('fewer_than_two_analysis_hops',inspect_word(self.word(1,1.021),self.clip))
        self.assertEqual(inspect_word(self.word(1,1.064),self.clip),[])
        self.assertIn('long_interval_low_vad_speech',inspect_word(self.word(1,5),self.clip))
        self.assertIn('non_positive_duration',inspect_word(self.word(1,1),self.clip))
        self.assertIn('outside_audio',inspect_word(self.word(10,11),self.clip))
        self.assertEqual(inspect_word({**self.word(1,1),'flag':'overlap'},self.clip),[])
    def test_manual_changes_are_guarded(self):
        w=self.word(1,1.02);entry={'source_sha256':'abc','action':'set','start':1.,'end':1.2}
        self.assertEqual(validate_override(entry,w,self.clip,'abc',[]),(1,1.2))
        for change in [{'source_sha256':'stale'},{'end':float('nan')},{'end':1.01},{'start':-1},{'end':12},{'action':'other'}]:
            with self.subTest(change=change),self.assertRaises(ValueError):validate_override({**entry,**change},w,self.clip,'abc',[])
        with self.assertRaises(ValueError):validate_override(entry,w,self.clip,'abc',[self.word(1.1,1.5)])
        self.assertIsNone(validate_override({'source_sha256':'abc','action':'exclude'},w,self.clip,'abc',[]))

if __name__=='__main__':unittest.main(verbosity=2)
