import copy
import hashlib
import json
from pathlib import Path
import tempfile
import time
import types
import unittest
from unittest.mock import patch
from scripts import resume_webprm_map_hosting as r
from scripts import prepare_webprm_map_recovery as builder


def fixture():
    native=dict(positive=dict(valid=True,score=1),negative=dict(valid=True,score=0))
    base=dict(schema=[dict(table_name='nodes')],constraints=[dict(constraint_name='nodes_pkey')],indexes=[dict(indexname='nodes_pkey')],migrations=['1'],counts=dict(current_nodes=0,current_ways=0,current_relations=0))
    rows=[]
    for phase in ('a-baseline','b-baseline','a-controls','b-controls'):
        row=dict(phase=phase,fresh_claim=phase.startswith('b-'),database_baseline=copy.deepcopy(base),schema_dump_sha256='a'*64,
                 database_marker_before=0,native_controls_run=phase.endswith('controls'),closure=dict(remaining_owned_processes={}))
        if phase.endswith('controls'):
            row.update(backend_pass=dict(tiles=True,geocode=True,routes=True),database_marker_after=1,database_marker_data=['owned-reset-control'],
                browser=dict(released_map_tasks=26,home=dict(status=200),initial_cookie_count=0,initial_origin_count=0,
                             storage_state_sha256=hashlib.sha256(phase.encode()).hexdigest(),ui_routes=dict(Foot='foot',Bicycle='bike'),
                             evaluator={k:copy.deepcopy(native) for k in ('task90_url_match','task87_program_html','task159_program_html')}))
        rows.append(row)
    return rows


class RecoveryTests(unittest.TestCase):
    def test_distinct_recovery_gate(self):
        gate=r.recovery_pass(fixture(),{'sha256':'base'},{'sha256':'base'})
        self.assertFalse(gate['two_fresh_states_verified'])
        self.assertFalse(gate['validation_states_are_production_baselines'])
        self.assertTrue(gate['resumed_A_verified'])
        self.assertTrue(gate['fresh_B_from_original_base_verified'])

    def test_full_schema_and_migrations_must_match(self):
        for key,value in (('schema_dump_sha256','b'*64),('database_baseline',dict(fixture()[0]['database_baseline'],migrations=['2']))):
            rows=fixture();rows[1][key]=value
            with self.assertRaises(r.h.Closed):r.recovery_pass(rows,1,1)

    def test_no_early_browser_or_marker(self):
        for key,value in (('browser',{}),('database_marker_after',1),('native_controls_run',True)):
            rows=fixture();rows[0][key]=value
            with self.assertRaises(r.h.Closed):r.recovery_pass(rows,1,1)

    def test_no_fresh_A_claim_or_base_change(self):
        rows=fixture();rows[0]['fresh_claim']=True
        with self.assertRaises(r.h.Closed):r.recovery_pass(rows,1,1)
        with self.assertRaises(r.h.Closed):r.recovery_pass(fixture(),1,2)

    def test_marker_data_and_empty_origins(self):
        for key,value in (('database_marker_data',['wrong']),('database_marker_before',1)):
            rows=fixture();rows[3][key]=value
            with self.assertRaises(r.h.Closed):r.recovery_pass(rows,1,1)
        rows=fixture();rows[3]['browser']['initial_origin_count']=1
        with self.assertRaises(r.h.Closed):r.recovery_pass(rows,1,1)

    def test_native_controls_and_phase_closure(self):
        rows=fixture();rows[2]['browser']['evaluator']['task87_program_html']['positive']['score']=0
        with self.assertRaises(r.h.Closed):r.recovery_pass(rows,1,1)
        rows=fixture();rows[0]['closure']['remaining_owned_processes']={'17':{}}
        with self.assertRaises(r.h.Closed):r.recovery_pass(rows,1,1)

    def test_charge_no_refund_no_double_count(self):
        self.assertEqual(r.storage_charge(100,80,20)['total_charged_bytes'],120)
        self.assertEqual(r.storage_charge(100,130,20)['total_charged_bytes'],150)
        self.assertEqual(r.storage_charge(100,130,20)['scratch_bytes'],50)
        with self.assertRaises(r.h.Closed):r.storage_charge(100,-1,0)

    def test_full_byte_fingerprint_and_symlink_nontraversal(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);(root/'a').write_text('one');(root/'outside').symlink_to('/does/not/exist')
            before=r.tree_fingerprint(root,time.time()+30)
            (root/'a').write_text('two')
            self.assertNotEqual(before['sha256'],r.tree_fingerprint(root,time.time()+30)['sha256'])

    def test_overlay_preserves_old_scripts_and_fail_closed_pid_collision(self):
        source=r.old.N/'map-hosting-validation-v4-vlm/source/scripts/webprm_map_recipe.py'
        namespace={};exec(compile(source.read_text(),str(source),'exec'),dict(__file__=str(source)),namespace)
        # Module imports have their own globals; only immutable script strings
        # are required by overlay_scripts.
        recipe=types.SimpleNamespace(**{k:namespace[k] for k in ('TILE_CTL','NOM_CTL','DB_CTL','WEB_CTL')})
        old=recipe.TILE_CTL;overlays=r.overlay_scripts(recipe,'job-a-baseline')
        self.assertEqual(recipe.TILE_CTL,old)
        self.assertIn('exit 91',overlays['tile'])
        self.assertIn('.webprm-preserved-job-a-baseline',overlays['tile'])
        self.assertNotIn('/webprm-phase',overlays['web'])
        self.assertNotIn('/tmp/migrate.log',overlays['web'])
        self.assertNotRegex(''.join(overlays.values()),r'(?m)(?:^|[;\s])rm\s')

    def test_frozen_accounting_patch_owns_prior_growth(self):
        text=(r.REPO/'openwebrl/webprm_cms_hosting.py').read_text();modified=builder.accounting_patch(text)
        self.assertIn('prior_scratch_paths=()',modified)
        self.assertIn('old_live-prior_scratch_bytes',modified)
        self.assertIn('storage_measure(),retained_bytes=',modified)
        self.assertIn("row.get('required_nodes')!='g007'",modified)
        self.assertNotEqual(text,modified)

    def test_exact_original_budget(self):
        self.assertEqual(839+13560,14399)
        self.assertLessEqual(839+13560,r.RESOURCES['total_seconds'])
        self.assertEqual(r.RESOURCES['max_allocations'],2)
        self.assertEqual(r.LIMITS['scratch_bytes'],384*2**30)
        self.assertEqual(r.LIMITS['api_usd'],0)

    def test_approval_required_before_runtime_effects(self):
        with patch.object(r.h,'validate',side_effect=r.h.Closed('no approval')),patch.object(r,'worker') as worker:
            with self.assertRaises(r.h.Closed):r.approved({'service_plan':{'sha256':'a'}},{})
            worker.assert_not_called()


if __name__=='__main__':unittest.main()
