"""Synthetic complete-grid arithmetic; no benchmark task payloads or providers."""
from copy import deepcopy
import unittest

from scripts.prepare_webprm_table5_campaign import cohort as make_cohort, ACTORS, SELECTORS, COUNTS
from openwebrl.webprm_table5_analysis import (
    aggregate_table5, digest, stratified_paired_bootstrap, validate_cohort, wilson_interval,
)


def fixture_cohort():
    tasks=[]
    for domain,n in COUNTS.items():
        for _ in range(n):
            i=len(tasks)
            tasks.append(dict(task_id=i,old_task_id=1000+i,sites=[domain],intent='Synthetic offline fixture'))
    result=make_cohort(tasks)
    result['protocol_sha256']='a'*64
    return result


def fixture_records(cohort, outcome=lambda _:0):
    cohort_sha=digest(cohort)
    result=[]
    for cell in cohort['cells']:
        score=outcome(cell)
        result.append(dict(cell_id=cell['cell_id'],released_task_id=cell['task_id'],
            actor_model=cell['actor'],selector=cell['selector'],domain=cell['domain'],
            task_config_sha256=cell['task_config_sha256'],tasks_sha256=cohort['source_sha256'],
            cohort_sha256=cohort_sha,protocol_sha256=cohort['protocol_sha256'],
            audit_verified=True,audit_sha256='b'*64,terminal=True,attempted=True,valid=True,
            success=score,native_success=score,effective_semantic_status='valid',native_semantic_status='valid'))
    return result


def invalidate(row,raw_native=None):
    row.update(valid=False,success=None,native_success=raw_native,
               effective_semantic_status='not_applicable',
               native_semantic_status='not_applicable' if raw_native is None else 'invalid')


def method(report, selector='gpt4omini',actor=ACTORS[0]):
    return next(x for x in report['methods'] if x['selector']==selector and x['actor_model']==actor)


def comparison(report,selector='gpt4omini',actor=ACTORS[0]):
    return next(x for x in report['comparisons'] if x['selector']==selector and x['actor_model']==actor)


class Table5AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.cohort=fixture_cohort()

    def aggregate(self,records):
        return aggregate_table5(self.cohort,records,bootstrap_draws=100,seed=17)

    def test_exact_unequal_domain_macro_is_not_pooled_rate(self):
        rows=fixture_records(self.cohort,lambda c:int(c['selector']!='no_search' and c['domain']=='shopping'))
        report=self.aggregate(rows);m=method(report)
        self.assertFalse(report['performance_rates_withheld'])
        self.assertEqual(m['counts']['planned'],155)
        self.assertEqual(m['counts']['effective_successes'],45)
        self.assertEqual(m['rates']['macro_success_percent'],20.)
        self.assertAlmostEqual(m['rates']['pooled_success_percent'],100*45/155)
        self.assertEqual(m['domains']['shopping']['rates']['denominator'],45)
        self.assertEqual(m['domains']['reddit']['rates']['denominator'],19)
        self.assertIsNotNone(m['domains']['reddit']['rates']['wilson95_percent'])
        paired=comparison(report)['primary_invalid_as_failure']
        self.assertEqual(paired['denominator'],155)
        self.assertEqual(paired['macro_gain_percentage_points'],20.)
        self.assertEqual(paired['macro_gain_bootstrap95_percentage_points'],[20.,20.])
        self.assertAlmostEqual(paired['pooled_gain_percentage_points'],100*45/155)

    def test_missing_or_unattempted_cell_blocks_all_performance_estimates(self):
        for unattempted in (False,True):
            rows=fixture_records(self.cohort,lambda _:1)
            if unattempted:
                invalidate(rows[-1]);rows[-1]['attempted']=False
            else:rows.pop()
            report=self.aggregate(rows)
            with self.subTest(unattempted=unattempted):
                self.assertEqual(report['counts']['planned'],1550)
                self.assertEqual(report['counts']['attempted'],1549)
                self.assertEqual(report['counts']['missing'],1)
                self.assertEqual(report['counts']['recorded_unattempted'],int(unattempted))
                self.assertTrue(report['performance_rates_withheld'])
                self.assertIn('incomplete_fixed_cohort',report['withheld_reasons'])
                for m in report['methods']:
                    self.assertIsNone(m['rates']['macro_success_percent'])
                    self.assertIsNone(m['rates']['pooled_success_percent'])
                    for d in m['domains'].values():
                        self.assertIsNone(d['rates']['success_percent'])
                        self.assertIsNone(d['rates']['valid_only_success_percent'])
                        self.assertIsNone(d['rates']['wilson95_percent'])
                for c in report['comparisons']:
                    self.assertIsNone(c['primary_invalid_as_failure']['macro_gain_percentage_points'])
                    self.assertIsNone(c['common_valid_secondary']['pooled_gain_percentage_points'])

    def test_empty_unbound_preparation_is_counts_only_not_completed(self):
        cohort=deepcopy(self.cohort);cohort['protocol_sha256']=None
        report=aggregate_table5(cohort,[],bootstrap_draws=100)
        self.assertEqual(report['counts']['missing'],1550)
        self.assertEqual(report['counts']['attempted'],0)
        self.assertIn('scientific_protocol_not_bound',report['withheld_reasons'])
        self.assertTrue(report['performance_rates_withheld'])

    def test_known_bad_or_unreviewed_effective_judge_prevents_rates(self):
        for semantic in ('invalid','unreviewed'):
            rows=fixture_records(self.cohort,lambda _:1)
            rows[0]['effective_semantic_status']=semantic
            report=self.aggregate(rows)
            self.assertEqual(report['counts']['effective_successes'],1550)
            self.assertEqual(report['counts']['effective_semantic_'+semantic],1)
            self.assertTrue(report['performance_rates_withheld'])
            self.assertIsNone(method(report)['rates']['pooled_success_percent'])

    def test_guarded_effective_zero_preserves_bad_raw_native_positive_counts(self):
        rows=fixture_records(self.cohort)
        for row in rows:row.update(native_success=1,native_semantic_status='invalid')
        report=self.aggregate(rows)
        self.assertFalse(report['performance_rates_withheld'])
        self.assertEqual(report['counts']['effective_successes'],0)
        self.assertEqual(report['counts']['native_successes_on_valid_episodes'],1550)
        self.assertEqual(report['counts']['changed_valid_verdicts'],1550)
        self.assertEqual(report['counts']['native_semantic_invalid'],1550)
        self.assertEqual(method(report)['rates']['macro_success_percent'],0)

    def test_invalid_attempts_stay_in_primary_denominator_not_common_valid(self):
        rows=fixture_records(self.cohort,lambda c:int(c['selector']=='gpt4omini'))
        for row in rows:
            if row['domain']=='map':
                if row['selector']=='gpt4omini' and row['released_task_id']%2==0:invalidate(row,raw_native=1)
                if row['selector']=='no_search' and row['released_task_id']%2==1:invalidate(row)
        report=self.aggregate(rows);m=method(report)
        self.assertEqual(m['counts']['attempted'],155)
        self.assertEqual(m['counts']['valid'],142)
        self.assertEqual(m['counts']['invalid'],13)
        self.assertEqual(m['counts']['missing'],0)
        self.assertEqual(m['counts']['raw_native_positive_verdicts_on_invalid_episodes'],13)
        self.assertEqual(m['rates']['macro_success_percent'],90)
        self.assertAlmostEqual(m['rates']['pooled_success_percent'],100*142/155)
        c=comparison(report)
        self.assertEqual(c['primary_invalid_as_failure']['denominator'],155)
        self.assertEqual(c['common_valid_secondary']['denominator'],129)
        self.assertEqual(c['common_valid_secondary']['pooled_gain_percentage_points'],100)
        self.assertIsNone(c['common_valid_secondary']['macro_gain_percentage_points'])
        self.assertIsNone(c['common_valid_secondary']['macro_gain_bootstrap95_percentage_points'])
        self.assertEqual(c['common_valid_secondary']['no_common_valid_domains'],['map'])

    def test_complete_infrastructure_failure_grid_withholds_headline_rates(self):
        rows=fixture_records(self.cohort)
        for row in rows:invalidate(row)
        report=self.aggregate(rows)
        self.assertTrue(report['terminal_grid_complete'])
        self.assertFalse(report['all_episodes_valid'])
        self.assertEqual(report['counts']['attempted'],1550)
        self.assertEqual(report['counts']['invalid'],1550)
        self.assertEqual(len(report['coverage_gaps']),50)
        self.assertTrue(report['performance_rates_withheld'])
        self.assertIn('domain_method_groups_without_valid_episodes',report['withheld_reasons'])
        self.assertIsNone(method(report)['rates']['macro_success_percent'])
        self.assertIn('not model-only',report['completion_limitation'])

    def test_zero_success_is_allowed_when_episodes_are_valid(self):
        report=self.aggregate(fixture_records(self.cohort))
        self.assertTrue(report['all_episodes_valid'])
        self.assertFalse(report['performance_rates_withheld'])
        self.assertEqual(report['coverage_gaps'],[])
        self.assertEqual(method(report)['rates']['macro_success_percent'],0)

    def test_paired_bootstrap_preserves_per_task_arm_correlation(self):
        # Marginal outcomes vary, but every arm agrees on each task. Independent
        # resampling of the two arms would incorrectly create nonzero intervals.
        rows=fixture_records(self.cohort,lambda c:c['task_id']%2)
        report=self.aggregate(rows)
        for c in report['comparisons']:
            for name in ('primary_invalid_as_failure','common_valid_secondary'):
                self.assertEqual(c[name]['macro_gain_percentage_points'],0)
                self.assertEqual(c[name]['macro_gain_bootstrap95_percentage_points'],[0.,0.])

    def test_seeded_bootstrap_and_record_order_are_reproducible(self):
        rows=fixture_records(self.cohort,lambda c:int((c['task_id']+SELECTORS.index(c['selector']))%3==0))
        self.assertEqual(self.aggregate(rows),self.aggregate(list(reversed(rows))))
        differences={d:[-1,0,1] for d in COUNTS}
        self.assertEqual(stratified_paired_bootstrap(differences,draws=100,seed=19),
                         stratified_paired_bootstrap(differences,draws=100,seed=19))

    def test_duplicate_foreign_or_mixed_identity_records_rejected(self):
        rows=fixture_records(self.cohort)
        with self.assertRaisesRegex(ValueError,'Duplicate'):self.aggregate(rows+[rows[0]])
        for field,value in [('cell_id','foreign'),('protocol_sha256','c'*64),('tasks_sha256','c'*64),
                            ('cohort_sha256','c'*64),('task_config_sha256','c'*64),('domain','foreign'),
                            ('actor_model',ACTORS[1] if rows[0]['actor_model']==ACTORS[0] else ACTORS[0]),
                            ('released_task_id',999),('selector','foreign')]:
            bad=deepcopy(rows);bad[0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):self.aggregate(bad)

    def test_unaudited_nonterminal_ambiguous_or_invalid_outcome_rejected(self):
        rows=fixture_records(self.cohort)
        for field,value in [('audit_verified',False),('audit_sha256',None),('terminal',False),
                            ('attempted',1),('valid',0),('success',True),('success',2),
                            ('success',None),('native_success',None),('effective_semantic_status','unknown')]:
            bad=deepcopy(rows);bad[0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):self.aggregate(bad)
        bad=deepcopy(rows);bad[0]['valid']=False
        with self.assertRaisesRegex(ValueError,'Invalid records'):self.aggregate(bad)

    def test_changed_planned_grid_and_per_task_identity_rejected(self):
        for mutation in ('missing','duplicate','mixed_task','counts','actors'):
            c=deepcopy(self.cohort)
            if mutation=='missing':c['cells'].pop()
            if mutation=='duplicate':c['cells'][-1]=c['cells'][0]
            if mutation=='mixed_task':c['cells'][0]['task_config_sha256']='d'*64
            if mutation=='counts':c['domain_counts']['reddit']=20
            if mutation=='actors':c['actors']=list(reversed(c['actors']))
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):validate_cohort(c)

    def test_marginal_wilson_interval_is_nonzero_at_observed_boundaries(self):
        zero=wilson_interval(0,19);all_success=wilson_interval(19,19)
        self.assertAlmostEqual(zero[0],0)
        self.assertGreater(zero[1],0)
        self.assertLess(all_success[0],100)
        self.assertAlmostEqual(all_success[1],100)
        self.assertIsNone(wilson_interval(0,0))
        with self.assertRaises(ValueError):wilson_interval(2,1)

    def test_public_output_contains_no_raw_task_or_audit_payload(self):
        rows=fixture_records(self.cohort)
        rows[0]['raw_task_private']='Do not expose this private payload'
        rows[0]['trajectory_private']=['secret']
        report=self.aggregate(rows)
        self.assertNotIn('Do not expose',str(report))
        self.assertNotIn('trajectory_private',str(report))
        self.assertIn('does not itself inspect raw trajectories',report['audit_scope'])


if __name__=='__main__':unittest.main()
