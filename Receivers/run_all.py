#!/usr/bin/env python3
"""
run_all.py — Concurrent test runner.
Launches N Webots+receiver pairs simultaneously, each processing one session.
Workers refill from the session queue until all sessions are complete.

Usage:
  python3 Receivers/run_all.py               # 2 workers, all 19 sessions
  python3 Receivers/run_all.py --workers 3   # 3 concurrent workers
  python3 Receivers/run_all.py --sessions baseline train_ycommand  # subset
"""

import argparse
import csv
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque

# ---------------------------------------------------------------------------
# Constants — edit here if your environment differs
# ---------------------------------------------------------------------------
WEBOTS_EXE      = '/Applications/Webots.app/Contents/MacOS/webots'
WORLD_FILE      = 'Worlds/wesDroneRL2.wbt'   # relative to project root
BASE_PORT       = 8080                        # worker 0 → 8080, 1 → 8081, …
STATUS_INTERVAL = 60                          # seconds between status prints
READY_TIMEOUT   = 30                          # seconds to wait for "Listening on port" signal
MAX_RESTARTS    = 3                           # max receiver crash restarts per session

# ---------------------------------------------------------------------------
# Resolve project root (two levels up from receivers/run_all.py)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECEIVERS_DIR = os.path.join(PROJECT_ROOT, 'Receivers')
RESULTS_BASE  = os.path.join(PROJECT_ROOT, 'Data', 'ALL_FINAL_ERROR_RESULTS')

# ---------------------------------------------------------------------------
# Import session list from all_receiver
# ---------------------------------------------------------------------------
sys.path.insert(0, RECEIVERS_DIR)
from all_receiver import SESSIONS, TEST_SEQUENCE  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def session_csv_path(session_name: str) -> str:
    return os.path.join(RESULTS_BASE, f'{session_name}_error_log.csv')


def session_is_complete(session_name: str) -> bool:
    """Return True if the CSV for this session already has test_index == 75 rows."""
    path = session_csv_path(session_name)
    if not os.path.exists(path):
        return False
    try:
        max_idx = 0
        with open(path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    idx = int(row.get('test_index', 0))
                    if idx > max_idx:
                        max_idx = idx
                except (ValueError, TypeError):
                    pass
        return max_idx >= len(TEST_SEQUENCE)
    except Exception:
        return False


def format_elapsed(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f'{h:02d}:{m:02d}:{sec:02d}'


# ---------------------------------------------------------------------------
# Worker class
# ---------------------------------------------------------------------------

class Worker:
    def __init__(self, worker_id: int):
        self.id        = worker_id
        self.port      = BASE_PORT + worker_id
        self.session   = None      # current session dict (or None if idle)
        self.recv_proc = None      # receiver Popen
        self.wbt_proc  = None      # Webots Popen
        self.restarts  = 0
        self._status_lines = []    # last ~15 lines captured from receiver stdout
        self._lock     = threading.Lock()
        self._reader_thread = None
        self._stop_reader   = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self, session: dict):
        """Launch receiver then Webots for the given session."""
        self.session = session
        self._stop_reader = False
        with self._lock:
            self._status_lines = []   # clear stale lines from previous run
        self._launch_receiver()

    def stop(self):
        """Terminate both subprocesses cleanly."""
        self._stop_reader = True
        self._kill_proc(self.wbt_proc,  'Webots')
        self._kill_proc(self.recv_proc, 'receiver')
        self.wbt_proc  = None
        self.recv_proc = None

    def is_done(self) -> bool:
        """True when the receiver process has exited."""
        if self.recv_proc is None:
            return True
        return self.recv_proc.poll() is not None

    def last_status(self) -> str:
        """Return the last meaningful status line from receiver stdout."""
        with self._lock:
            lines = list(self._status_lines)
        # Prefer [step= lines, then TEST lines, then whatever we have
        for line in reversed(lines):
            if '[step=' in line or 'TEST ' in line:
                return line.strip()
        return lines[-1].strip() if lines else '(no output yet)'

    def status_snapshot(self) -> list:
        """Return up to 3 recent interesting lines."""
        with self._lock:
            lines = list(self._status_lines)
        interesting = [l.strip() for l in lines
                       if 'TEST ' in l or '[step=' in l or 'complete' in l.lower()]
        return interesting[-3:] if interesting else ([lines[-1].strip()] if lines else [])

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _launch_receiver(self):
        """Start the receiver subprocess and wait for its ready signal."""
        cmd = [
            sys.executable,
            '-u',
            os.path.join(RECEIVERS_DIR, 'all_receiver.py'),
            '--session', self.session['name'],
            '--port',    str(self.port),
        ]
        self.recv_proc = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Start background reader immediately so we don't block the pipe
        self._reader_thread = threading.Thread(
            target=self._read_stdout, daemon=True)
        self._reader_thread.start()

        # Wait for "Listening on port P" signal
        ready = self._wait_for_ready()
        if not ready:
            print(f'[WORKER {self.id}] WARNING: receiver did not signal ready '
                  f'within {READY_TIMEOUT}s — launching Webots anyway')

        self._launch_webots()

    def _launch_webots(self):
        """Start the Webots subprocess with port injected via environment."""
        world_abs = os.path.join(PROJECT_ROOT, WORLD_FILE)
        cmd = [
            WEBOTS_EXE,
            '--no-rendering',
            '--batch',
            world_abs,
        ]
        env = os.environ.copy()
        env['WEBOTS_CONTROLLER_PORT'] = str(self.port)
        self.wbt_proc = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _wait_for_ready(self) -> bool:
        """Block until receiver prints 'Listening on port' or timeout."""
        target = f'Listening on port {self.port}'
        deadline = time.time() + READY_TIMEOUT
        while time.time() < deadline:
            with self._lock:
                for line in self._status_lines:
                    if target in line:
                        return True
            time.sleep(0.2)
        return False

    def _read_stdout(self):
        """Background thread: read receiver stdout into _status_lines."""
        try:
            for line in self.recv_proc.stdout:
                if self._stop_reader:
                    break
                with self._lock:
                    self._status_lines.append(line)
                    if len(self._status_lines) > 50:
                        self._status_lines = self._status_lines[-30:]
        except Exception:
            pass

    @staticmethod
    def _kill_proc(proc, label: str):
        if proc is None:
            return
        if proc.poll() is None:
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
            except Exception as e:
                print(f'    [kill {label}] {e}')


# ---------------------------------------------------------------------------
# Status display
# ---------------------------------------------------------------------------

def print_status(workers: list, completed: list, queue: deque, start_time: float):
    elapsed = time.time() - start_time
    remaining = len(queue)
    n_workers = len(workers)
    active = [w for w in workers if w.session is not None]

    bar = '═' * 63
    print(f'\n{bar}')
    print(f'  run_all.py  —  elapsed: {format_elapsed(elapsed)}'
          f'  |  {n_workers} workers  |  queue: {remaining} remaining')
    print(bar)

    for w in workers:
        if w.session is None:
            print(f'  WORKER {w.id}  [port {w.port}]  (idle)')
            continue
        print(f'  WORKER {w.id}  [port {w.port}]  {w.session["name"]}')
        for line in w.status_snapshot():
            print(f'    › {line}')
        if not w.status_snapshot():
            print(f'    › (starting up…)')

    print(bar)
    if completed:
        print(f'  Completed: {", ".join(completed)}')
    else:
        print(f'  Completed: [none yet]')
    print(bar)
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Concurrent Webots test runner.')
    parser.add_argument('--workers', type=int, default=2,
                        help='Number of concurrent workers (default 2)')
    parser.add_argument('--sessions', nargs='+', metavar='NAME',
                        help='Run only these named sessions (default: all 19)')
    args = parser.parse_args()

    # Build session list
    all_names = {s['name'] for s in SESSIONS}
    if args.sessions:
        unknown = [n for n in args.sessions if n not in all_names]
        if unknown:
            print(f'Unknown session name(s): {", ".join(unknown)}')
            print(f'Valid names: {", ".join(sorted(all_names))}')
            sys.exit(1)
        sessions_to_run = [s for s in SESSIONS if s['name'] in args.sessions]
    else:
        sessions_to_run = list(SESSIONS)

    # Skip already-complete sessions
    queue_list = [s for s in sessions_to_run if not session_is_complete(s['name'])]
    skipped    = [s['name'] for s in sessions_to_run if session_is_complete(s['name'])]
    if skipped:
        print(f'Skipping already-complete sessions: {", ".join(skipped)}')

    if not queue_list:
        print('All sessions already complete. Nothing to do.')
        return

    queue     = deque(queue_list)
    workers   = [Worker(i) for i in range(args.workers)]
    completed = []
    start_time = time.time()

    # Graceful shutdown handler
    def _shutdown(sig, frame):
        print('\n\nInterrupted — stopping all workers …')
        for w in workers:
            w.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Seed workers with initial sessions
    for w in workers:
        if queue:
            w.start(queue.popleft())

    last_status_time = 0.0  # force immediate first print

    print(f'\nStarting {len(workers)} workers for {len(queue_list)} sessions …')

    while any(w.session is not None for w in workers):
        time.sleep(2)

        for w in workers:
            if w.session is None:
                continue
            if not w.is_done():
                continue

            # Receiver exited
            rc = w.recv_proc.returncode if w.recv_proc else -1

            if rc != 0 and w.restarts < MAX_RESTARTS:
                print(f'[WORKER {w.id}] receiver crashed (rc={rc}), '
                      f'restart {w.restarts + 1}/{MAX_RESTARTS} — '
                      f'resuming {w.session["name"]}')
                w.stop()
                w.restarts += 1
                session = w.session
                w.session = None
                w.start(session)
            else:
                if rc != 0:
                    print(f'[WORKER {w.id}] receiver failed after {w.restarts} restart(s) '
                          f'— giving up on {w.session["name"]}')
                else:
                    completed.append(w.session['name'])
                    print(f'[WORKER {w.id}] session complete: {w.session["name"]}')

                w.stop()
                w.restarts = 0
                w.session  = None

                if queue:
                    w.start(queue.popleft())

        if time.time() - last_status_time >= STATUS_INTERVAL:
            print_status(workers, completed, queue, start_time)
            last_status_time = time.time()

    print_status(workers, completed, queue, start_time)
    print(f'\nAll sessions complete. Total time: {format_elapsed(time.time() - start_time)}')


if __name__ == '__main__':
    main()
