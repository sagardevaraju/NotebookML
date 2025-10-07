"""Helpers to abstract DataFrame engine selection."""
from __future__ import annotations

import contextlib
import os
from typing import Any, Dict, Optional


class BackendManager:
    """Factory for DataFrame operations across pandas-like engines.

    Parameters
    ----------
    engine:
        One of ``"pandas"``, ``"modin"``, or ``"dask"``.
    modin_engine:
        Backend to use when ``engine="modin"``. Defaults to ``"ray"``.
    dask_kwargs:
        Optional keyword arguments forwarded to :class:`dask.distributed.Client`.
    """

    def __init__(
        self,
        engine: str = "pandas",
        *,
        modin_engine: str = "ray",
        dask_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.engine = engine.lower()
        self._modin_engine = modin_engine
        self._dask_kwargs = dask_kwargs or {"processes": False}
        self.client = None
        self.frame_namespace = self._initialise_namespace()

    def _initialise_namespace(self):
        if self.engine == "pandas":
            import pandas as pd  # type: ignore

            return pd
        if self.engine == "modin":
            os.environ.setdefault("MODIN_ENGINE", self._modin_engine)
            if os.environ["MODIN_ENGINE"] == "ray":
                import ray

                if not ray.is_initialized():
                    ray.init(ignore_reinit_error=True)
            elif os.environ["MODIN_ENGINE"] == "dask":
                import modin.config as modin_config

                modin_config.Engine.put("dask")
            import modin.pandas as pd  # type: ignore

            return pd
        if self.engine == "dask":
            import dask.dataframe as dd  # type: ignore
            from dask.distributed import Client

            if Client.current() is None:
                self.client = Client(**self._dask_kwargs)
            else:
                self.client = Client.current()
            return dd
        raise ValueError(f"Unsupported engine: {self.engine}")

    # ------------------------------------------------------------------
    # I/O helpers
    def read_csv(self, path: str, **kwargs):
        return self.frame_namespace.read_csv(path, **kwargs)

    def read_parquet(self, path: str, **kwargs):
        return self.frame_namespace.read_parquet(path, **kwargs)

    def to_csv(self, df, path: str, **kwargs):
        if self.engine == "dask":
            df.compute().to_csv(path, **kwargs)
        elif self.engine == "modin":
            df.to_csv(path, **kwargs)
        else:
            df.to_csv(path, **kwargs)

    def to_parquet(self, df, path: str, **kwargs):
        if self.engine == "dask":
            df.compute().to_parquet(path, **kwargs)
        else:
            df.to_parquet(path, **kwargs)

    # ------------------------------------------------------------------
    def to_pandas(self, df):
        if self.engine == "dask":
            return df.compute()
        if self.engine == "modin":
            return df._to_pandas() if hasattr(df, "_to_pandas") else df.to_pandas()
        return df

    def close(self) -> None:
        if self.client:
            with contextlib.suppress(Exception):
                self.client.close()
            self.client = None


__all__ = ["BackendManager"]
