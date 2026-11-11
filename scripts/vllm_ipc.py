#!/usr/bin/env python3
"""Private short directories for vLLM's Unix sockets, kept on the data disk."""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import tempfile

MAX_DIR_BYTES = 60  # Leave room for vLLM's UUID and socket suffixes (Linux limit 107).


def prefix(tag):
    return 'sa-ipc-' + hashlib.sha256(tag.encode()).hexdigest()[:10] + '-'


def create(preferred, tag, root=None):
    preferred = Path(preferred).expanduser().absolute()
    preferred.mkdir(parents=True, exist_ok=True)
    candidates = [Path(root).expanduser().absolute()] if root else [preferred, *preferred.parents]
    for base in candidates:
        if base == Path('/') or len(os.fsencode(str(base / (prefix(tag) + 'x' * 8)))) > MAX_DIR_BYTES:
            continue
        if root:
            base.mkdir(parents=True, exist_ok=True)
        if not base.is_dir() or not os.access(base, os.W_OK | os.X_OK):
            continue
        try:
            return Path(tempfile.mkdtemp(prefix=prefix(tag), dir=str(base)))
        except OSError:
            if root:
                raise
    raise RuntimeError('No short writable IPC directory. Set SPECAGENT_IPC_ROOT to a writable short data-disk path, e.g. /data/your-user/ipc.')


def cleanup(path, tag):
    path = Path(path)
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or not path.is_dir() or path.stat().st_uid != os.getuid() or not path.name.startswith(prefix(tag)):
        raise ValueError(f'Refusing to remove an IPC directory not owned by this run: {path}')
    shutil.rmtree(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['create', 'cleanup'])
    parser.add_argument('--tag', required=True)
    parser.add_argument('--tmp-root')
    parser.add_argument('--root')
    parser.add_argument('--path')
    args = parser.parse_args()
    if args.action == 'create':
        if not args.tmp_root:
            parser.error('--tmp-root is required for create')
        print(create(args.tmp_root, args.tag, args.root or None))
    else:
        if not args.path:
            parser.error('--path is required for cleanup')
        cleanup(args.path, args.tag)


if __name__ == '__main__':
    main()
