#!/usr/bin/env python3
"""Reserve free TCP ports across concurrent launchers owned by this user.

Advisory locks live on the host, not in a checkout or experiment work directory.
Sockets are checked with bind (including bound-but-not-listening sockets). Locks
remain held while models load; services themselves must bind their own sockets.
"""
import argparse
import errno
import fcntl
import os
from pathlib import Path
import signal
import socket
import time


def lock_directory():
    base = Path('/dev/shm') if Path('/dev/shm').is_dir() else Path('/tmp')
    directory = base / f'specagent-ports-{os.getuid()}'
    directory.mkdir(mode=0o700, exist_ok=True)
    info = directory.lstat()
    if directory.is_symlink() or not directory.is_dir() or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError(f'Port lock directory must be private and owned by this user: {directory}')
    return directory


def port_available(port):
    # vLLM / sandbox / Redis use wildcard addresses. Test both address families,
    # without SO_REUSEADDR, so an existing listener is never treated as reusable.
    sockets = []
    try:
        for family, address in [(socket.AF_INET, '0.0.0.0'), (socket.AF_INET6, '::')]:
            try:
                listener = socket.socket(family, socket.SOCK_STREAM)
            except OSError as exc:
                if family == socket.AF_INET6 and exc.errno in {errno.EAFNOSUPPORT, errno.EPROTONOSUPPORT}:
                    continue
                raise
            sockets.append(listener)
            if family == socket.AF_INET6:
                listener.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            try:
                listener.bind((address, port))
            except OSError as exc:
                if family == socket.AF_INET6 and exc.errno == errno.EADDRNOTAVAIL:
                    continue  # Host has IPv6 disabled.
                if exc.errno == errno.EADDRINUSE:
                    return False
                raise
        return True
    finally:
        for listener in sockets:
            listener.close()


def reserve(requests, offset, tries, directory=None):
    directory = directory or lock_directory()
    locks, selected = [], {}
    try:
        for name, start in requests:
            if not 1 <= start <= 65535:
                raise ValueError(f'{name} port out of range: {start}')
            if name in selected:
                raise ValueError(f'Duplicate service name: {name}')
            for attempt in range(tries):
                port = start + attempt * offset
                if port > 65535:
                    break
                if port in selected.values():
                    continue
                fd = os.open(directory / f'{port}.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
                keep = False
                try:
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        continue
                    if not port_available(port):
                        continue
                    locks.append(fd)
                    selected[name] = port
                    keep = True
                    break
                finally:
                    if not keep:
                        os.close(fd)
            if name not in selected:
                raise ValueError(f'No free {name} port from {start}, offset={offset}, tries={tries}')
        return selected, locks
    except BaseException:
        for fd in locks:
            os.close(fd)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', action='append', required=True, help='service=starting-port')
    parser.add_argument('--offset', type=int, default=100)
    parser.add_argument('--tries', type=int, default=50)
    parser.add_argument('--parent', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.offset < 1 or args.tries < 1:
        parser.error('offset and tries must be positive')
    requests = []
    for item in args.port:
        name, value = item.split('=', 1)
        requests.append((name, int(value)))
    selected, locks = reserve(requests, args.offset, args.tries)
    running = True

    def stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        temporary = args.output.with_suffix(f'.{os.getpid()}.tmp')
        temporary.write_text(''.join(f'{name} {port}\n' for name, port in selected.items()))
        temporary.replace(args.output)
        # Parent identity via getppid also detects a dead launcher without
        # mistaking a recycled PID for that parent.
        while running and os.getppid() == args.parent:
            time.sleep(0.1)
    finally:
        for fd in locks:
            os.close(fd)
        # Never unlink lock files: doing so would let waiters lock two inodes.


if __name__ == '__main__':
    main()
