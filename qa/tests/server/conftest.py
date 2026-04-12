from contextlib import contextmanager


class FakeCursor:
    def __init__(self, conn):
        self.conn = conn
        self.current = None
        self.lastrowid = conn.lastrowid

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=None):
        self.conn.executed.append((sql, params))
        if self.conn.responses:
            self.current = self.conn.responses.pop(0)
        else:
            self.current = None
        self.lastrowid = self.conn.lastrowid

    def fetchone(self):
        if isinstance(self.current, dict):
            return self.current
        return {}

    def fetchall(self):
        if isinstance(self.current, list):
            return self.current
        return []


class FakeConn:
    def __init__(self, responses=None, lastrowid=1):
        self.responses = list(responses or [])
        self.executed = []
        self.lastrowid = lastrowid
        self.commits = 0

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        self.commits += 1


@contextmanager
def with_conn(conn):
    yield conn
