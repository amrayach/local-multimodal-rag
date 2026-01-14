from dataclasses import dataclass, field
from pathlib import Path


def _default_base() -> Path:
    return Path("/home/ammer/local_multimodal_rag/data/mmrag")


@dataclass(frozen=True)
class Settings:
    base_dir: Path = field(default_factory=_default_base)

    api_host: str = "0.0.0.0"
    api_port: int = 3001

    ui_host: str = "0.0.0.0"
    ui_port: int = 8081

    @property
    def docs_dir(self) -> Path:
        return self.base_dir / "docs"

    @property
    def index_dir(self) -> Path:
        return self.base_dir / "index"

    @property
    def cache_dir(self) -> Path:
        return self.base_dir / "cache"


settings = Settings()
