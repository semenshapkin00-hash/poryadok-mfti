"""One local worker; startup verifies weights and optionally seeds fictional data only."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import uuid

from container_models import prepare

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    if os.environ.get('NAVIGATOR_REQUIRE_MODELS', '1') == '1':
        prepare(ROOT / 'models')  # Verification only; startup never downloads.
    data = Path(os.environ.get('NAVIGATOR_DATA_DIR', ROOT / 'data')).resolve()
    data.mkdir(parents=True, exist_ok=True, mode=0o700)
    (ROOT / '.cache').mkdir(exist_ok=True)
    public = os.environ.get('NAVIGATOR_PUBLIC_DEMO') == '1'
    seed = os.environ.get('NAVIGATOR_SEED_DEMO') == '1'
    if public or seed:
        import app
        if public and not getattr(app, 'PUBLIC_DEMO', False):
            raise RuntimeError('Public startup requires the application public-demo policy.')
        if public:
            for path in (data / 'documents').glob('*.json'):
                if json.loads(path.read_text()).get('metadata', {}).get('synthetic') is not True:
                    raise RuntimeError('Public demo data directory contains a non-synthetic record.')
        if seed:
            manifest = json.loads((ROOT / 'demo/manifest.json').read_text())
            if manifest.get('synthetic') is not True or manifest.get('official_competition_data') is not False:
                raise RuntimeError('Only the declared fictional demo may be seeded.')
            entries = manifest['documents']
            identifiers = {item['id']: uuid.uuid5(uuid.NAMESPACE_URL, 'spn-poryadok-demo:' + item['id']).hex for item in entries}
            for item in entries:
                path = (ROOT / 'demo' / item['path']).resolve()
                if not path.is_relative_to(ROOT / 'demo/documents') or path.suffix not in {'.md', '.xlsx'} or item['metadata'].get('synthetic') is not True:
                    raise RuntimeError('Invalid fictional demo entry.')
                metadata = dict(item['metadata'])
                metadata['supersedes'] = [identifiers.get(value, value) for value in metadata.get('supersedes', [])]
                app.persist_document(item.get('name', path.name), path.read_bytes(), metadata, identifiers[item['id']])
    port = int(os.environ.get('PORT', '8769'))
    if not 1024 <= port <= 65535:
        raise ValueError('PORT must be between 1024 and 65535.')
    os.execv(sys.executable, [sys.executable, '-m', 'uvicorn', 'app:app', '--host', '0.0.0.0', '--port', str(port), '--workers', '1'])


if __name__ == '__main__':
    main()
