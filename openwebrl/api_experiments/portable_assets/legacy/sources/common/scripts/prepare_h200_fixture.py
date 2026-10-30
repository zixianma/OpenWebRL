"""Create explicitly synthetic rewards for a bounded training plumbing test."""
import asyncio
import json
import os
from urllib.parse import quote
from pathlib import Path
import torch
from playwright.async_api import async_playwright
from transformers import AutoTokenizer
from slime.utils.types import Sample


async def main():
    root = Path(os.environ['OPENWEBRL_RUNTIME_ROOT']) / 'runs' / 'fixture'
    root.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=['--no-sandbox'])
        page = await browser.new_page(viewport={'width': 448, 'height': 448})
        await page.set_content('<button onclick="this.textContent=\'DONE\'">CONTINUE</button>')
        await page.get_by_role('button', name='CONTINUE').click()
        assert await page.get_by_role('button').inner_text() == 'DONE'
        await page.screenshot(path=str(root / 'browser.png'))
        await browser.close()
    print('Real Chromium launch, navigation, click and screenshot PASS', flush=True)
    tok = AutoTokenizer.from_pretrained('/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT')
    prompt = tok.apply_chat_template([{'role': 'user', 'content': 'What is two plus two?'}], tokenize=True, add_generation_prompt=True)
    samples = []
    for i in range(4):
        response = 'Four.' if i % 2 else 'Five.'
        ids = tok.encode(response, add_special_tokens=False)
        s = Sample(index=i, group_index=i//2, tokens=prompt+ids, response=response,
                   response_length=len(ids), reward=float(i % 2), status=Sample.Status.COMPLETED)
        samples.append(s.to_dict())
    torch.save({'samples': samples}, root / 'rollout.pt')
    (root / 'prompts.jsonl').write_text(json.dumps({'prompt': [{'role': 'user', 'content': 'What is two plus two?'}]}) + '\n')
    browser_task = {'prompt': [{'role': 'user', 'content': 'Report the button text.'}], 'metadata': {'task_id': 'local_smoke/12345',
        'start_url': 'data:text/html,' + quote('<button>DONE</button>'),
        'task': 'Report the text on the button, then finish the task.'}}
    (root / 'browser-prompts.jsonl').write_text(json.dumps(browser_task) + '\n')
    print('Synthetic text-only replay fixture saved:', root, flush=True)


if __name__ == '__main__':
    asyncio.run(main())
