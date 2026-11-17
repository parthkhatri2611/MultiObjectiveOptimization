"""Build the pinned official llama.cpp server and serve the pinned GGUF on loopback."""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
import subprocess
import tarfile
import urllib.request
from .records import atomic_json


def build(workspace):
    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    sources = json.loads((Path(__file__).parent / 'resolved_sources.json').read_text())
    revision = sources['llama']['revision']
    source = workspace / ('llama.cpp-' + revision)
    if not source.exists():
        archive = workspace / 'llama.cpp.tar.gz'
        urllib.request.urlretrieve(f'https://github.com/ggml-org/llama.cpp/archive/{revision}.tar.gz', archive)
        with tarfile.open(archive) as tar:
            tar.extractall(workspace, filter='data')
    # Fail before compiling if CUDA is not actually available.
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('No CUDA GPU in this notebook session; enable a Kaggle GPU first')
    major, minor = torch.cuda.get_device_capability(0)
    arch = str(major * 10 + minor)
    build_dir = source / 'build'
    # Notebook images can expose a versioned driver without the development symlink
    # that CMake expects. Use the real loaded driver, never the toolkit stub.
    import ctypes
    ctypes.CDLL('libcuda.so.1')
    driver_paths = [line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                    if '/libcuda.so' in line and '/' in line.split()[-1]]
    if not driver_paths:
        raise RuntimeError('CUDA driver loaded but its library path could not be determined')
    driver_library = driver_paths[0]
    subprocess.run(['cmake', '-S', str(source), '-B', str(build_dir), '-DGGML_CUDA=ON',
                    f'-DCMAKE_CUDA_ARCHITECTURES={arch}', '-DLLAMA_CURL=OFF',
                    f'-DCUDA_cuda_driver_LIBRARY={driver_library}',
                    '-DLLAMA_BUILD_TESTS=OFF', '-DLLAMA_BUILD_EXAMPLES=OFF', '-DLLAMA_BUILD_SERVER=ON'], check=True)
    subprocess.run(['cmake', '--build', str(build_dir), '--config', 'Release', '-j', '2',
                    '--target', 'llama-server'], check=True)
    binary = build_dir / 'bin/llama-server'
    if not binary.exists():
        raise FileNotFoundError('Pinned llama-server binary was not built')
    return binary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--context', type=int, default=16384)
    args = parser.parse_args()
    workspace = Path(args.workspace).resolve()
    metadata_path = Path(args.metadata).resolve()
    saved_bin = metadata_path.parent / 'runtime/bin'
    if (saved_bin / 'llama-server').exists() and metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        for name, digest in previous.get('runtime_files_sha256', {}).items():
            if hashlib.sha256((saved_bin / name).read_bytes()).hexdigest() != digest:
                raise ValueError('Saved server runtime checksum mismatch: ' + name)
        binary = saved_bin / 'llama-server'
    else:
        built = build(workspace)
        shutil.copytree(built.parent, saved_bin, dirs_exist_ok=True)
        binary = saved_bin / 'llama-server'
    from huggingface_hub import hf_hub_download
    pinned = json.loads((Path(__file__).parent / 'resolved_sources.json').read_text())
    model = pinned['model']
    paths = [hf_hub_download('Qwen/Qwen2.5-7B-Instruct-GGUF', filename=name,
                             revision=model['revision'], cache_dir=str(workspace / 'weights')) for name in model['files']]
    hashes = {}
    for name, path in zip(model['files'], paths):
        digest = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        hashes[name] = digest.hexdigest()
    metadata = {'model_repo': 'Qwen/Qwen2.5-7B-Instruct-GGUF', 'model_revision': model['revision'],
                'files_sha256': hashes, 'server_revision': pinned['llama']['revision'],
                'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
                'runtime_files_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                         for p in saved_bin.iterdir() if p.is_file()},
                'quantization': 'Q4_K_M', 'context': args.context,
                'server_version': subprocess.run([str(binary), '--version'], capture_output=True, text=True).stdout}
    command = [str(binary), '-m', paths[0], '--alias', 'rm-qwen25-7b-q4',
               '--host', '127.0.0.1', '--port', '8765', '-c', str(args.context),
               '-ngl', '99', '--parallel', '1', '--jinja', '--seed', '20261114',
               '--threads', '2', '--no-context-shift']
    metadata['command'] = command
    atomic_json(args.metadata, metadata)
    subprocess.run(command, check=True, env={**os.environ, 'CUDA_VISIBLE_DEVICES': '0'})


if __name__ == '__main__':
    main()
