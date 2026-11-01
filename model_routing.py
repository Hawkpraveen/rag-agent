"""Deterministic task routing. Model names come only from server configuration."""
import re

MODES = {'auto', 'fast', 'thorough'}


def select_route(question, sources, mode, models, installed):
    if mode not in MODES:
        raise ValueError('Use auto, fast, or thorough model mode.')
    text = question.lower()
    context_size = sum(len(s.get('text', '')) for s in sources)
    document_count = len({s.get('source') for s in sources})
    if mode != 'auto':
        tier, reason = mode, f'{mode.capitalize()} mode selected'
    elif re.search(r'\b(code|coding|debug|function|sql|python|javascript|typescript|algorithm)\b', text):
        tier, reason = 'code', 'Question asks about code'
    elif re.search(r'\b(compare|comparison|differences?|conflicts?|contradictions?|analy[sz]e|reasoning|tradeoffs?|evaluate)\b|step.by.step|in.depth|in detail', text):
        tier, reason = 'thorough', 'Question asks for comparison or detailed analysis'
    elif context_size > 6500 or document_count > 3 or len(question) > 600:
        tier, reason = 'thorough', 'Large question or document context'
    else:
        tier, reason = 'fast', 'Routine lookup or short overview'

    def resolve(name):
        return next((m for m in installed if m == name or m == name + ':latest'), None)

    chosen = resolve(models[tier])
    if chosen is None:
        chosen = resolve(models['fast']) or resolve(models['default'])
        if chosen is None:
            raise RuntimeError('No configured chat model is installed in Ollama.')
        reason += '; preferred model unavailable, using installed fallback'
    return {'mode': mode, 'model': chosen, 'reason': reason}
 