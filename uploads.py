"""Validate and retain uploaded source files so citations remain resolvable."""
import hashlib
import os
from pathlib import Path
import re
import tempfile
from urllib.parse import unquote

MAX_FILE_BYTES = 10 * 1024 * 1024

def upload_name(encoded):
    name = unquote(encoded or '')
    if (not name or len(name) > 180 or any(c in name for c in ('/', '\\'))
            or any(ord(c) < 32 for c in name)
            or Path(name).suffix.lower() not in {'.txt', '.md', '.pdf'}):
        raise ValueError('Attach a TXT, Markdown (.md), or PDF file with a valid filename.')
    return re.sub(r'[^\w. -]', '_', name).strip(' .')

def save_upload(root, name, content):
    if not content:
        raise ValueError('The attached file is empty.')
    if len(content) > MAX_FILE_BYTES:
        raise ValueError('File exceeds the 10 MB limit.')
    digest = hashlib.sha256(content).hexdigest()
    # The digest separates distinct uploads and reuses the same path for retries.
    directory = Path(root) / digest
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / name
    with tempfile.NamedTemporaryFile(dir=directory, delete=False) as handle:
        temp_path = Path(handle.name)
        try:
            handle.write(content)
        except BaseException:
            handle.close()
            temp_path.unlink(missing_ok=True)
            raise
    try:
        os.replace(temp_path, target)
    finally:
        temp_path.unlink(missing_ok=True)
    return target
