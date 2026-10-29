from bootstrap import ROOT,CONFIG
import tempfile,unittest,json,types,hashlib
from pathlib import Path
from claude_budget import ReservationBudget
from claude_model import ClaudeModel,PRICING
from common_judge import judge
from episode_metrics import receipt_metrics
class Integration(unittest.TestCase):
 def test_model_and_judge_receipts_preserve_separate_pools(self):
  with tempfile.TemporaryDirectory(dir='/tmp')as temp:
   p=Path(temp);ep=p/'episode';ep.mkdir();image=ep/'final.png';image.write_bytes(b'offline-fixture-no-live-image')
   class Reply:
    def model_dump(self):return {'model':'gpt-5.6-terra','choices':[{'finish_reason':'stop','message':{'content':'status: failure'}}],'usage':{'prompt_tokens':100,'completion_tokens':5,'prompt_tokens_details':{'cached_tokens':0}}}
   class Fake:
    chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=lambda **kw:Reply()))
   pool=ReservationBudget(p/'judge',2.5,1100);self.assertEqual(judge(Fake(),pool,'offline task',[],image,ep),0);self.assertEqual(pool.snapshot()['calls'],1);self.assertEqual(pool.snapshot()['reserved_calls'],0)
   def actor(payload):return {'id':'offline','type':'message','role':'assistant','model':'claude-sonnet-5-5','stop_reason':'end_turn','content':[{'type':'text','text':'done'}],'usage':{'input_tokens':100,'output_tokens':10,'cache_read_input_tokens':0,'cache_creation_input_tokens':0}}
   modelpool=ReservationBudget(p/'model',5,2000);model=ClaudeModel('sonnet_single',ep,actor,modelpool,PRICING['claude-sonnet-5-5'],openai_api_key='offline-no-network',response_mode='run_command_tool');out=model.query([{'role':'system','content':'offline'},{'role':'user','content':'offline task'}]);self.assertTrue(out['extra']['done']);self.assertEqual(modelpool.snapshot()['calls'],1)
   stats=receipt_metrics(ep,PRICING['claude-sonnet-5-5']);self.assertEqual(stats['metric_errors'],[]);self.assertEqual(stats['model_excluding_judge']['calls'],1);self.assertEqual(stats['by_role']['judge']['calls'],1);self.assertAlmostEqual(stats['model_excluding_judge']['received_cost_usd'],.0003)
 def test_missing_usage_keeps_full_reserve_in_episode_metrics(self):
  with tempfile.TemporaryDirectory(dir='/tmp')as temp:
   p=Path(temp);(p/'requests').mkdir();(p/'requests/claude-unknown.json').write_text(json.dumps({'role':'candidate_0','status':'usage_unknown','reserved_usd':.5}));m=receipt_metrics(p,PRICING['claude-sonnet-5-5']);self.assertEqual(m['model_excluding_judge']['charged_or_reserved_usd'],.5);self.assertEqual(m['model_excluding_judge']['unknown_usage_calls'],1)
 def test_model_and_judge_prices_and_original_task_manifest(self):
  self.assertEqual(CONFIG['model_pricing'],PRICING);self.assertEqual(CONFIG['pricing']['judge_input'],1.1);self.assertEqual(CONFIG['pricing']['judge_output'],4.4);self.assertEqual(hashlib.sha256((ROOT/'tasks.json').read_bytes()).hexdigest(),'fb9fb44e30f2933391ec8489990c377978a1066cfc694b754a10042ce8faaab1')
if __name__=='__main__':unittest.main()
