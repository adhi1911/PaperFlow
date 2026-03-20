from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # Project paths
    project_root: Path = Path(__file__).parent.parent.parent
    data_dir: Path = Path(__file__).parent.parent.parent / "data"
    raw_papers_dir: Path = Path(__file__).parent.parent.parent / "data" / "raw_papers"
    
    # File paths
    manifest_path: Path = Path(__file__).parent.parent.parent / "data" / "papers_manifest.json"
    chunks_file: Path = Path(__file__).parent.parent.parent / "data" / "chunks.jsonl"
    
    # Chunking configuration
    chunk_size: int = 1000
    chunk_overlap: int = 200
    chunk_min_size: int = 100
    
    # Text splitting separators
    separators: list = ["\n\n", "\n", " ", ""]
    
    # Document processing
    pdf_extensions: set = {".pdf"}
    
    # Logging
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Create singleton instance
settings = Settings()

# Ensure directories exist
settings.data_dir.mkdir(parents=True, exist_ok=True)
settings.raw_papers_dir.mkdir(parents=True, exist_ok=True)

