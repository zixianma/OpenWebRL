"""Reproduce frozen GPT-high BrowserCode policies; dry run by default."""
import sys
sys.dont_write_bytecode = True
from common import main

if __name__ == '__main__':
    sys.exit(main('gpt'))
