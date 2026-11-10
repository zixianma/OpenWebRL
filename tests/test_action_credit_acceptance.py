"""Saved-file auditing with CPU browser/provider fakes; no physical acceptance.

The native pending-batch prerequisite is mocked. All episode, journal, source,
transfer, verdict, and allocation files below are actually read and verified.
"""
from dataclasses import replace
from pathlib import Path
import json
import time
import unittest
from unittest.mock import patch

import test_action_credit_collect as collection_fixture
import test_action_credit_judge as judge_fixture
from test_action_credit_state import sha
from openwebrl.action_credit import ResponseGraph, digest
from openwebrl.action_credit_acceptance import audit_allocation, audit_saved_collection
from openwebrl.action_credit_collect import FrozenActor
from openwebrl.action_credit_judge import JudgeHTTPProtocol, MeteredJudgeClient
from openwebrl.action_credit_native_controller import _hash, _json, _write, validator_source_sha256
from openwebrl.action_credit_state import CampaignManifest, CampaignStore
from openwebrl.action_credit_worker import _settle_terminal
from openwebrl.controlled_sft_worker import AttemptJournal


class AcceptanceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        fixture = collection_fixture.CollectionTests()
        fixture.setUp()
        self.addCleanup(fixture.temp.cleanup)
        self.f = fixture
        self.root = Path(fixture.temp.name)
        source_root = Path(__file__).resolve().parents[1]
        names = {p.relative_to(source_root).as_posix() for p in (source_root/'openwebrl').glob('action_credit*.py')}
        names |= {'openwebrl/'+name+'.py' for name in ('generate_browser','reward_browser','arm_branch_protocol',
            'arm_branch_browser','rl_recipe','controlled_sft_worker','controlled_browser_server','response_format')}
        source = dict(source_root=str(source_root),files={name:_hash(source_root/name) for name in sorted(names)})
        self.source_ref = _write(self.root/'sources.json',source)
        data = fixture.store.manifest.to_dict()
        self.approval = dict(status='approved',campaign_id=data['campaign_id'],model='fixture-judge',
            api_microdollars_cap=100,api_requests_cap=100,evidence='CPU fake; not external approval')
        self.protocol = JudgeHTTPProtocol('fixture-judge',digest(self.approval),sha('CPU input bound'),32,50,1_000_000,1_000_000)
        data['code_revision_sha256'] = digest(source)
        data['science']['judge'].update(http_protocol_sha256=self.protocol.sha256,
            canonical_reward_source_sha256=source['files']['openwebrl/reward_browser.py'])
        fixture.store = CampaignStore.create(self.root/'audited-ledger',CampaignManifest.from_dict(data))
        self.store = fixture.store
        self.prior = dict(path=str(self.root/'prior-inventory.json'),sha256=sha('CPU prerequisite inventory'))
        self.rng = _write(self.root/'rng.json',dict(policy_sha256=fixture.contract.policy_sha256,
            graph_contract_sha256=fixture.contract.sha256,previous_checkpoint_inventory=self.prior))
        freeze = _write(self.root/'freeze.json',dict(all_eight_transfer_rpcs_awaited=True,
            policy_sha256=fixture.contract.policy_sha256,graph_contract_sha256=fixture.contract.sha256,
            weight_version='1',endpoints=['fixture://actor'],rng_reference=self.rng,checkpoint_inventory=self.prior))
        fixture.actor = FrozenActor(fixture.contract.policy_sha256,'1',self.rng['sha256'],freeze['sha256'],digest('fixture://actor'))
        async def version(): return '1'
        fixture.version = version
        state=self.store.snapshot(); state['driver_freezes']={freeze['sha256']:freeze}; self.store._write(state)
        self.episode = await fixture.run_episode(collection_fixture.NativeFake(),
            reserved_cost=dict(gpu_seconds=0,api_requests=0,api_microdollars=0,browser_dispatches=1))
        self.journal=AttemptJournal(Path(self.episode.artifact_directory)/'judge')
        async def provider(**payload): return judge_fixture.Reply()
        client=MeteredJudgeClient(store=self.store,parent_attempt_id='attempt1',protocol=self.protocol,
            approval_receipt=self.approval,provider_create=provider,provider_max_retries=0,
            input_token_upper_bound=lambda payload:6,input_bound_protocol_sha256=self.protocol.input_bound_protocol_sha256,
            journal=self.journal)
        await client.create(model='fixture-judge',messages=[{'role':'user','content':'CPU fixture'}])
        result=dict(status='valid',reward=1.,native_values=[1.],source_sha256=source['files']['openwebrl/reward_browser.py'],
            native_reward=dict(judge=1.,valid=True,judge_timeout=False,judge_text='SUCCESS',
                               judge_prompt_variant='action_history',format=1.,combined=1.))
        _settle_terminal(self.store,self.episode,result,journal=self.journal,tree_terminal=False)
        turn=self.episode.turns[0]
        edge=replace(turn.edge,old_log_probs=(-1.,)*len(turn.edge.response_ids),old_log_prob_source='frozen_actor_teacher_forced')
        leaf=self.episode.leaf(status='valid',reward=1.)
        graph=ResponseGraph(fixture.contract,(edge,),(leaf,),(leaf.leaf_id,))
        self.payload=dict(graph=graph,rows={edge.edge_id:turn.row},tensor_payloads={edge.edge_id:turn.tensors},
            replay_receipts={edge.edge_id:turn.replay_receipt},leaf_to_draw={leaf.leaf_id:self.episode.draw_id},optimizer_rng_artifact=self.rng)
        self.pointer=dict(file_sha256=sha('CPU prerequisite pending'),binding_sha256=sha('CPU prerequisite binding'),
            ppo_schedule_sha256=sha('CPU prerequisite schedule'))
        self.native_ref=_write(self.root/'native.json',dict(verification_kind='cpu_mock',passed=True,
            manifest_sha256=self.store.manifest.sha256,validator_source_sha256=validator_source_sha256(),
            **{k:True for k in ('actual_model','full_response_masks','tp_dp_gradient_parity','ppo2_windows',
                              'native_checkpoint_restore','per_rank_rng_restore','native_controller_integration')}))
        self.accounting=self.accounting_ref()

    def accounting_ref(self,*,raw='123|COMPLETED|2|cpu=64,gres/gpu=8,mem=960G|\n',attempt_ids=('123',),elapsed=2):
        index=len(list(self.root.glob('accounting-*.json')))
        raw_path=self.root/f'sacct-{index}.txt'; raw_path.write_text(raw)
        ledger=_write(self.root/f'allocations-{index}.json',dict(manifest_sha256=self.store.manifest.sha256,
            approved_gpu_seconds_cap=100,attempts=[dict(job_id=x) for x in attempt_ids]))
        return _write(self.root/f'accounting-{index}.json',dict(schema_version=1,manifest_sha256=self.store.manifest.sha256,
            observed_unix=time.time(),approved_gpu_seconds_cap=100,attempt_ledger=ledger,
            scheduler_records=[dict(path=str(raw_path),sha256=_hash(raw_path))],
            attempts=[dict(job_id='123',gpus=8,elapsed_seconds=elapsed,state='COMPLETED')],gpu_seconds_used=8*elapsed))

    def audit(self,**overrides):
        kwargs=dict(source_inventory_ref=self.source_ref,native_acceptance_ref=self.native_ref,judge_protocol=self.protocol,
            allocation_accounting_ref=self.accounting,diagnostic_only=True)
        kwargs.update(overrides)
        with patch('openwebrl.action_credit_acceptance.load_scored_pending',return_value=(self.pointer,self.payload,None)):
            return _json(audit_saved_collection(self.store,self.f.contract,self.pointer,**kwargs))

    async def test_saved_records_are_verified_and_remain_cpu_diagnostics(self):
        result=self.audit()
        self.assertEqual(result['verification_kind'],'cpu_mock')
        self.assertTrue(result['diagnostic_only'])
        self.assertEqual((result['attempts_checked'],result['edges_checked'],result['http_attempts_checked']),(1,1,1))
        self.assertEqual(result['charged_resources'],dict(gpu_seconds=0,api_requests=1,api_microdollars=6,browser_dispatches=1))
        self.assertEqual(result['allocation_gpu_seconds_upper_bound'],16)

    async def test_cpu_evidence_never_authorizes_production(self):
        with self.assertRaisesRegex(ValueError,'CPU diagnostics cannot authorize production'):
            self.audit(diagnostic_only=False)

    async def test_saved_episode_media_corruption_rejected(self):
        media=next(Path(self.episode.artifact_directory).glob('*.png'))
        media.write_bytes(b'changed after collection')
        with self.assertRaises((ValueError,RuntimeError)):
            self.audit()

    async def test_scored_graph_cannot_change_sampled_mask_or_context(self):
        graph=self.payload['graph']
        for changes in (dict(sampled_mask=(True,True,True,True)),dict(input_ids=(10,666))):
            with self.subTest(changes=changes):
                self.payload['graph']=replace(graph,edges=(replace(graph.edges[0],**changes),))
                with self.assertRaisesRegex(ValueError,'changed a selected saved response'):
                    self.audit()
        self.payload['graph']=graph

    async def test_rehashed_wrong_terminal_reward_is_rejected(self):
        state=self.store.snapshot(); ref=state['terminal_verdicts']['attempt1']
        record=_json(ref); record['verdict']['native_reward']['judge']=0.
        new_ref=_write(self.root/'contradictory-verdict.json',record)
        state['terminal_verdicts']['attempt1'].update(new_ref)
        state['attempts']['attempt1']['completion']['outcome']['artifact_sha256']=new_ref['sha256']
        self.store._write(state)
        with self.assertRaisesRegex(ValueError,'terminal binary reward'):
            self.audit()

    async def test_http_undercharge_is_rejected(self):
        state=self.store.snapshot(); row=next(iter(state['subrequests'].values()))
        row['charged']['api_microdollars']=0
        self.store._write(state)
        with self.assertRaisesRegex(ValueError,'cost does not match'):
            self.audit()

    async def test_actor_transfer_evidence_required(self):
        state=self.store.snapshot(); state['driver_freezes']={}; self.store._write(state)
        with self.assertRaisesRegex(ValueError,'driver transfer receipt'):
            self.audit()

    async def test_scheduler_raw_evidence_and_all_attempts_are_required(self):
        for changes in (dict(attempt_ids=('123','122')),dict(raw='123|COMPLETED|2|gres/gpu=4|\n'),
                        dict(elapsed=0),dict(raw='123|COMPLETED|2|gres/gpu=8|\n123|COMPLETED|2|gres/gpu=8|\n')):
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                audit_allocation(self.accounting_ref(**changes),self.store.manifest)
        with self.assertRaisesRegex(ValueError,'stale'):
            audit_allocation(self.accounting,self.store.manifest,now=time.time()+400)


if __name__=='__main__': unittest.main()
