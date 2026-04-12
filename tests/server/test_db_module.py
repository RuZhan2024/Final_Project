import pytest
import types

from applications.backend import db as db_mod


def test_require_env_raises_when_missing(monkeypatch):
    monkeypatch.delenv("DB_HOST", raising=False)
    with pytest.raises(RuntimeError):
        db_mod._require_env("DB_HOST", default=None)


def test_get_conn_optional_yields_none_on_failure(monkeypatch):
    def _boom():
        raise RuntimeError("db error")

    monkeypatch.setattr(db_mod, "get_conn", _boom)
    with db_mod.get_conn_optional() as conn:
        assert conn is None


def test_get_conn_import_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name.startswith("pymysql"):
            raise ImportError("no pymysql")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    with pytest.raises(RuntimeError):
        with db_mod.get_conn():
            pass


def test_get_conn_success_close_error_and_optional_success(monkeypatch):
    class _Conn:
        def commit(self):
            return None

        def close(self):
            raise RuntimeError("close failed")

    fake_mod = types.SimpleNamespace(connect=lambda **_k: _Conn())
    fake_cursors = types.SimpleNamespace(DictCursor=object)

    monkeypatch.setitem(__import__("sys").modules, "pymysql", fake_mod)
    monkeypatch.setitem(__import__("sys").modules, "pymysql.cursors", fake_cursors)

    with db_mod.get_conn() as _conn:
        pass

    with db_mod.get_conn_optional() as conn:
        assert conn is not None
