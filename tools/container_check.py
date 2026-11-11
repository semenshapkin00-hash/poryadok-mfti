"""Static container checks, localhost health, or an explicit synthetic inference smoke."""
from __future__ import annotations

import argparse
import ast
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def static():
    for name in ['container_models.py', 'container_entrypoint.py', 'container_check.py']:
        ast.parse((ROOT / 'tools' / name).read_text())
    dockerfile = (ROOT / 'Dockerfile').read_text()
    assert not re.search(r'^COPY\s+(?:--\S+\s+)*\.\s', dockerfile, re.M)
    patterns = [line for line in (ROOT / '.dockerignore').read_text().splitlines() if line and not line.startswith('#')]
    assert patterns[0] == '**'
    assert all('!' + name not in patterns for name in ['data/', 'reference/', 'models/', '.venv/', '.env', 'node_modules/'])
    manifest = json.loads((ROOT / 'demo/manifest.json').read_text())
    assert manifest['synthetic'] is True and manifest['official_competition_data'] is False
    for entry in manifest['documents']:
        path = (ROOT / 'demo' / entry['path']).resolve()
        assert path.is_relative_to(ROOT / 'demo/documents') and path.suffix in {'.md', '.xlsx'} and entry['metadata']['synthetic'] is True
        assert '!demo/' + entry['path'] in patterns and not path.is_symlink()
    assert '!demo/documents/*.md' not in patterns
    pins = [line.strip() for line in (ROOT / 'requirements-lock.txt').read_text().splitlines() if line.strip() and not line.lstrip().startswith('#')]
    assert pins and all(re.fullmatch(r'[A-Za-z0-9_.-]+==[^\s;]+', line) for line in pins)
    docker = shutil.which('docker') or str(Path.home() / '.docker/bin/docker')
    compose = 'not available'
    if Path(docker).is_file():
        result = subprocess.run([docker, 'compose', '-f', str(ROOT / 'compose.yaml'), 'config', '--format', 'json'], capture_output=True, text=True, timeout=20, check=True)
        cfg = json.loads(result.stdout)
        service = cfg['services']['app']
        assert service['ports'][0]['host_ip'] == '127.0.0.1' and service['read_only'] is True
        assert cfg['networks']['offline']['internal'] is True
        assert all(volume['type'] == 'volume' for volume in service['volumes'])
        assert service['build']['target'] == 'runtime'
        compose = 'valid (no daemon required)'
    return {'static_checks': True, 'locked_python_packages': len(pins), 'compose': compose, 'image_build_tested': False}


def health():
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open('http://127.0.0.1:' + os.environ.get('PORT', '8769') + '/api/health', timeout=4) as response:
        report = json.load(response)
    expected_mode = 'demo_server' if os.environ.get('NAVIGATOR_PUBLIC_DEMO') == '1' else 'local'
    assert report['status'] == 'ok' and report['processing'] == expected_mode
    if os.environ.get('NAVIGATOR_REQUIRE_MODELS', '1') == '1':
        assert report['models']['semantic']['available'] and report['models']['ocr']['available']
    return {'health': 'ok', 'models': report['models']}


def smoke():
    sys.path.insert(0, str(ROOT))
    from container_models import prepare
    prepare(ROOT / 'models')
    from app_services.semantic import embedding
    vector = embedding('Учебный пример: сотрудник оформляет вымышленную командировку.')
    assert len(vector) == 384 and all(math.isfinite(float(value)) for value in vector)
    assert abs(sum(float(value) ** 2 for value in vector) - 1) < 1e-4
    from PIL import Image, ImageDraw, ImageFont
    from navigator_engine.models import DocumentInput
    from app_services.ocr import extract_pages
    image = Image.new('RGB', (1000, 180), 'white')
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 52)
    ImageDraw.Draw(image).text((30, 50), 'PUBLIC DEMO 123', fill='black', font=font)
    content = io.BytesIO()
    image.save(content, format='PNG')
    result = extract_pages(DocumentInput('container-smoke', 'synthetic.png', content.getvalue(), {'synthetic': True}))
    assert result['offline'] is True and len(result['pages']) == 1 and result['pages'][0]['lines']
    return {'synthetic_smoke': True, 'semantic_dimensions': 384, 'ocr_pages': 1, 'external_api_used': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--static', action='store_true')
    mode.add_argument('--health', action='store_true')
    mode.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    print(json.dumps(static() if args.static else health() if args.health else smoke(), ensure_ascii=False))
