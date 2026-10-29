from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parent
CONFIG=json.loads((ROOT/'runtime-config.json').read_text())
sys.path[:0]=[str(ROOT),CONFIG['cuawright_source'],CONFIG['frozen_support_source']]
