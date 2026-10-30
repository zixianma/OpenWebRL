"""Exercise the repository's subprocess browser transport without a GPU or judge."""
import asyncio
from urllib.parse import quote
import yaml
from openwebrl.env.local_process_env import create_local_process_env


async def main():
    cfg = yaml.safe_load(open('openwebrl/env/config.yaml'))
    env = await create_local_process_env({'startup_timeout_secs': 60})
    try:
        await env.initialize('local-smoke', {'start_url': 'data:text/html,' + quote('<h1>Pipeline test</h1>')}, cfg)
        obs, info = await env.reset()
        assert obs['screenshot'] is not None
        print('Repository LocalProcessWebEnv reset and screenshot PASS', flush=True)
    finally:
        await env.exit()
    assert env.proc.returncode is not None
    print('Repository LocalProcessWebEnv subprocess cleanup PASS', flush=True)


if __name__ == '__main__':
    asyncio.run(main())
