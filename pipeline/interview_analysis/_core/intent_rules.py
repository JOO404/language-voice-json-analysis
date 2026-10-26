"""Small documented corrections to the original five-label intent classifier.

Scores are rule evidence, not probabilities. No participant/task-specific labels.
"""
import re,sys
from pathlib import Path
from . import detect_intent as legacy
VERSION='methodology-reconciliation-1'
AGREE=re.compile(r'(?<![가-힣])동의(?:합니다|해요|해|한다|한다는|하고|할게요|하겠습니다|하는\s*(?:편|것))(?=[\s.!?,]|$)')
DISAGREE=re.compile(r'동의\s*(?:하지\s*(?:않|못)|할\s*수\s*없|안\s*|못\s*)|(?:안|못)\s*동의')

def detect_intent(text,prosody=None):
    text=text.strip()
    if not text:return 'unknown',{},[],'low'
    _,scores,signals,_=legacy.detect_intent(text,None)
    # The statement score is a fallback, not an opposing lexical observation.
    scores['statement']=0.
    # '...하고 그래요' and '뭐라고 그래요' describe habits/reporting; they
    # are not stand-alone agreement. Do not promote this substring match.
    if ('affirmation','긍정 표현: 그래요') in signals and not re.search(r'(?:^|[.!?]\s*)그래요(?:[.!?]|$)',text):
        signals.remove(('affirmation','긍정 표현: 그래요'));scores['affirmation']-=1
    # A direct short wh-question need not retain a '?' after transcription.
    last_clause=re.split(r'[.!?]',text.rstrip('.!?'))[-1].strip()
    if len(last_clause.split())<=8 and re.match(r'^(?:왜|언제|어디|누가|누구|무엇|뭐)\b',last_clause) and not re.search(r'(?:지\b|라고|다고|것\s*같)',last_clause) and re.search(r'(?:니|냐|늦었어|왔어|했어|됐어|어디야|뭐야)$',last_clause):
        scores['question']+=2;signals.append(('question','의문 종결어미: -직접 의문사 질문'))
    # Missing Korean agreement/negative agreement forms; negative scope wins.
    disagreement=DISAGREE.search(text)
    agreement=AGREE.search(text)
    if disagreement:
        scores['negation']+=2
        signals.append(('negation','부정 표현: 동의 부정'))
    elif agreement and not re.search(r'동의[^.!?]*아니',text):
        scores['affirmation']+=2
        signals.append(('affirmation','긍정 표현: 동의'))
    # 그렇죠 is already an explicit agreement marker. Ambiguous -죠 alone
    # must not outweigh it without an actual question mark.
    if not text.endswith('?') and text.rstrip('.! ,;').endswith('그렇죠'):
        for label,evidence in list(signals):
            if label=='question' and evidence=='의문 종결어미: -죠':
                signals.remove((label,evidence));scores['question']-=2
    # A lone question word within a narrative is weak evidence, not a question.
    candidates=[k for k in ['question','affirmation','negation','hesitation'] if scores[k]>0 and (k!='question' or scores[k]>=2)]
    if candidates:
        primary=max(candidates,key=lambda k:scores[k])
        strength=scores[primary]
        tied=sum(scores[k]==strength for k in candidates)>1
        confidence='low' if tied or strength<=1 else 'high' if strength>=3 else 'medium'
    else:
        primary='statement';scores['statement']=1.;confidence='low'
    # Retain weak acoustic evidence for audit only. It never reverses explicit
    # lexical evidence or turns a narrative into a question on its own.
    if prosody and prosody.get('prosody_reliable') and prosody.get('words'):
        slope=prosody['words'][-1].get('f0_slope')
        if slope is not None:
            if slope>=legacy.F0_RISE_THRESHOLD:signals.append(('question',f'마지막 단어 F0 상승: +{slope:.0f} Hz/s'))
            elif slope<=legacy.F0_FALL_THRESHOLD:signals.append(('statement',f'마지막 단어 F0 하강: {slope:.0f} Hz/s'))
    return primary,scores,signals,confidence

def detect_speech_mode(text):
    mode,signals=legacy.detect_speech_mode(text)
    if mode=='direct' and re.search(r'얘기(?:를)?\s*할\s*(?:것|거)',text):
        return 'reported',[('reported',"가정형 발화: '…얘기할 것/거'")]
    return mode,signals
