"""Verified Folio help, resolved before document retrieval or model calls."""
import re


UPLOAD_HELP = (
    'Click **Add documents** in your library, or **Attach** beside the chat box. '
    'Select a PDF, TXT, or Markdown file up to 10 MB. '
    'Wait until Folio says the file is ready for questions.'
)


def app_help_answer(question):
    text = re.sub(r'[^a-z0-9 ]', ' ', question.lower())
    text = ' '.join(text.split())
    # Explicit document/source questions belong to RAG, even when about uploads.
    if re.search(r'\b(?:according to|in the document|in my document|does .* say|documents? say|manual|handbook|policy)\b', text):
        return None
    # Anchor to a help request: merely mentioning an upload is not app help.
    request = re.search(r'^(?:please )?(?:how (?:do|can|should) (?:i|we) |how to |where (?:do|can) (?:i|we) |where (?:is|s) (?:the )?|can you (?:help me |show me how to )?|help me (?:to )?)', text)
    action = re.search(r'\b(?:upload|attach|add|import)\b.*\b(?:doc|docs|document|documents|file|files|pdf|txt|markdown)\b', text)
    if request and action:
        return {'answer': UPLOAD_HELP, 'answer_type': 'app_help', 'sources': []}
    return None
