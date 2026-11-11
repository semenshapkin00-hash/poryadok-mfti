"""Explicit public-weight setup and offline verification. No credentials or corpus inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import urllib.request

SPECS = {
    'semantic-minilm': {
        'repo': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        'revision': 'e8f8c211226b894fcb81acc59f3b34ba3efd5f42',
        'base': 'https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/resolve/e8f8c211226b894fcb81acc59f3b34ba3efd5f42/',
        'files': {
            'model_qint8_arm64.onnx': ('onnx/model_qint8_arm64.onnx', '783fea82d71a58179b830a4dbd2d58447e640609e98eedf9ffa12622d375a672'),
            'tokenizer.json': ('tokenizer.json', '2c3387be76557bd40970cec13153b3bbf80407865484b209e655e5e4729076b8'),
            'tokenizer_config.json': ('tokenizer_config.json', '5036ea374ffedd706e3bef33e2e0d6953cb868ef8a490e76e32ba0faa37a6b9b'),
            'config.json': ('config.json', '6300193cb75e01cf80c96decef7187dfb33094d97cc1490b7ead6ff134476e4e'),
        },
    },
    'tessdata': {
        'repo': 'tesseract-ocr/tessdata_fast',
        'revision': '87416418657359cb625c412a48b6e1d6d41c29bd',
        'base': 'https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/87416418657359cb625c412a48b6e1d6d41c29bd/',
        'files': {
            'rus.traineddata': ('rus.traineddata', 'e16e5e036cce1d9ec2b00063cf8b54472625b9e14d893a169e2b0dedeb4df225'),
            'eng.traineddata': ('eng.traineddata', '7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2'),
        },
    },
}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify(target):
    for folder, spec in SPECS.items():
        directory = Path(target) / folder
        manifest = json.loads((directory / 'manifest.json').read_text())
        if manifest.get('repo') != spec['repo'] or manifest.get('revision') != spec['revision']:
            raise ValueError('Unexpected public model revision: ' + folder)
        entries = {entry['name']: entry for entry in manifest['files']}
        if set(entries) != set(spec['files']):
            raise ValueError('Unexpected public model file list: ' + folder)
        for name, (_, expected) in spec['files'].items():
            path = directory / name
            if entries[name]['sha256'] != expected or path.stat().st_size != entries[name]['bytes'] or digest(path) != expected:
                raise ValueError('Public model checksum mismatch: ' + folder + '/' + name)
    return {'verified': True, 'network_used': False, 'public_files': 6}


def prepare(target, *, download=False, copy_from=None):
    target = Path(target)
    if not download and copy_from is None:
        return verify(target)
    target.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(target).free < 5 * 1024**3 + 256 * 1024**2:
        raise RuntimeError('Model setup must preserve at least 5 GiB free disk.')
    for folder, spec in SPECS.items():
        directory = target / folder
        directory.mkdir(exist_ok=True)
        files = []
        for name, (remote, expected) in spec['files'].items():
            destination = directory / name
            if not destination.is_file() or digest(destination) != expected:
                temporary = destination.with_name(name + '.part')
                try:
                    source = urllib.request.urlopen(spec['base'] + remote, timeout=90) if download else (Path(copy_from) / folder / name).open('rb')
                    total = 0
                    with source, temporary.open('wb') as output:
                        while chunk := source.read(1024**2):
                            total += len(chunk)
                            if total > 256 * 1024**2:
                                raise ValueError('Unexpected public model size.')
                            if shutil.disk_usage(target).free - len(chunk) < 5 * 1024**3:
                                raise RuntimeError('Model setup reached the disk reserve.')
                            output.write(chunk)
                    if digest(temporary) != expected:
                        raise ValueError('Public download/copy checksum mismatch: ' + name)
                    os.replace(temporary, destination)
                finally:
                    temporary.unlink(missing_ok=True)
            files.append({'name': name, 'file': 'models/' + folder + '/' + name, 'bytes': destination.stat().st_size, 'sha256': expected})
        manifest = {'repo': spec['repo'], 'revision': spec['revision'], 'license': 'Apache-2.0', 'files': files, 'token_used': False, 'user_data_uploaded': False}
        temporary = directory / 'manifest.json.part'
        temporary.write_text(json.dumps(manifest, indent=2) + '\n')
        os.replace(temporary, directory / 'manifest.json')
    result = verify(target)
    result['network_used'] = bool(download)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--download', action='store_true', help='Fetch only six public SHA-pinned model assets.')
    group.add_argument('--copy-from', type=Path, help='Copy only the six declared files from an existing offline model directory.')
    parser.add_argument('--target', type=Path, default=Path('/app/models'))
    args = parser.parse_args()
    print(json.dumps(prepare(args.target, download=args.download, copy_from=args.copy_from)))
