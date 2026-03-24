from pathlib import Path 
from typing import Dict, Any, Optional
import yaml 
import json 
import logging 
from backend.utils.path_utils import get_relative_path

logger = logging.getLogger(__name__)

class PresetRegistry: 
    def __init__(self, 
                preset_dir: Path | str = None,
                project_root: Optional[Path] = None
            ): 
        if preset_dir is None: 
            preset_dir = Path(__file__).parent.parent / "config" / "presets"

        self.preset_dir = Path(preset_dir)
        self.project_root = project_root
        self._presets_cache: Dict[str, Dict[str, Any]] = {}
        self._load_presets()

    def _load_presets(self): 
        if not self.preset_dir.exists():
            relative_path = get_relative_path(self.preset_dir, self.project_root)
            logger.warning(f"Preset directory not found: {relative_path}")
            return

        for preset_file in self.preset_dir.glob("*.yaml"): 
            preset_name = preset_file.stem 
            try: 
                with open(preset_file, "r", encoding = "utf-8") as f: 
                    config = yaml.safe_load(f)
                    self._presets_cache[preset_name] = config
                    logger.info(f"Loaded preset: {preset_name}")
            except Exception as e:
                logger.error(f"Error loading preset {preset_name}: {e}")

    def get(self,
            preset_name: str
            ) -> Dict[str, Any]: 
        if preset_name not in self._presets_cache:
            logger.error(f"Preset not found: {preset_name}")
            raise ValueError(f"Preset not found: {preset_name}")

        return self._presets_cache[preset_name]

    def list_presets(self) -> list: 
        return list(self._presets_cache.keys())
    
    def reload(self): 
        self._presets_cache.clear()
        self._load_presets()
        logger.info("Reloaded presets")