from pathlib import Path
import ast,copy,hashlib,importlib.util,json,os,sys,time,types,unittest
ROOT=Path(__file__).resolve().parents[1]
R=ROOT
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from openwebrl.action_credit_checkpoint_semantics import native_checkpoint_scheduler

def make(style='constant',wdstyle='constant'):
    optimizer=types.SimpleNamespace(param_groups=[dict(lr_mult=1.,wd_mult=1.,max_lr=1e-6,min_lr=0.,default_config=True)])
    s=OptimizerParamScheduler(optimizer,init_lr=0.,max_lr=1e-6,min_lr=0.,lr_warmup_steps=0,
        lr_decay_steps=100000,lr_decay_style=style,start_wd=.1,end_wd=.1 if wdstyle=='constant' else .2,
        wd_incr_steps=100000,wd_incr_style=wdstyle)
    return s

def actual_wrapper():
    path=ROOT/'slime/backends/megatron_utils/checkpoint.py'
    tree=ast.parse(path.read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='load_checkpoint')
    ns=dict(Path=Path,get_args=lambda:types.SimpleNamespace(load=str(R)),
            _is_dir_nonempty=lambda p:True,_is_megatron_checkpoint=lambda p:True)
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),ns)
    return ns

class SchedulerProof(unittest.TestCase):
    def test_actual_additive_bug(self):
        s=make();s.step(2560);saved=copy.deepcopy(s.state_dict());s.step(256)
        s.load_state_dict(saved)
        self.assertEqual(s.num_steps,5376)
        self.assertNotEqual(s.state_dict(),saved)
    def test_cold_exact(self):
        source=make();source.step(2560);saved=source.state_dict();s=make()
        native_checkpoint_scheduler(s).load_state_dict(saved)
        self.assertEqual(s.state_dict(),saved)
    def test_warm_exact(self):
        s=make();s.step(2560);saved=copy.deepcopy(s.state_dict());s.step(256)
        native_checkpoint_scheduler(s).load_state_dict(saved)
        self.assertEqual(s.state_dict(),saved)
    def test_repeated_exact(self):
        s=make();s.step(2560);saved=s.state_dict();proxy=native_checkpoint_scheduler(s)
        for _ in range(3):proxy.load_state_dict(saved);self.assertEqual(s.state_dict(),saved)
    def test_zero_saved_counter(self):
        s=make();saved=s.state_dict();s.step(2560)
        native_checkpoint_scheduler(s).load_state_dict(saved);self.assertEqual(s.num_steps,0)
    def test_legacy_num_iters(self):
        s=make();s.step(768);saved=s.state_dict();saved['num_iters']=saved.pop('num_steps')
        native_checkpoint_scheduler(s).load_state_dict(saved);self.assertEqual(s.num_steps,768)
    def test_none_preserved(self):self.assertIsNone(native_checkpoint_scheduler(None))
    def test_invalid_saved_counter_does_not_mutate(self):
        for value in (None,-1,2.,True):
            s=make();s.step(512);saved=s.state_dict();saved['num_steps']=value
            with self.assertRaises(ValueError):native_checkpoint_scheduler(s).load_state_dict(saved)
            self.assertEqual(s.num_steps,512)
    def test_failure_propagates_and_restores_counter(self):
        s=make();s.step(512);saved=s.state_dict();saved.pop('min_lr')
        with self.assertRaises(KeyError) as caught:native_checkpoint_scheduler(s).load_state_dict(saved)
        self.assertEqual(s.num_steps,512)
        self.assertIn('do not continue training',caught.exception.__notes__[0])
    def test_complete_nonlinear_scheduler_and_group_rates(self):
        s=make('cosine','linear');s.step(2560)
        saved=copy.deepcopy(s.state_dict());groups=copy.deepcopy(s.optimizer.param_groups)
        s.step(256)
        native_checkpoint_scheduler(s).load_state_dict(saved)
        self.assertEqual(s.state_dict(),saved);self.assertEqual(s.optimizer.param_groups,groups)
    def test_wrapper_only_actual_native_load_calls_are_corrected(self):
        ns=actual_wrapper();s=make();s.step(2560);saved=s.state_dict();s.step(256)
        def native(**kwargs):kwargs['opt_param_scheduler'].load_state_dict(saved);return 101,0
        ns['_load_checkpoint_megatron']=native
        self.assertEqual(ns['load_checkpoint']([],None,s,{},False),(101,0))
        self.assertEqual(s.state_dict(),saved)
    def test_skipped_load_preserves_scheduler(self):
        # no_load_optim/finetune/release native branches never invoke scheduler.load_state_dict.
        ns=actual_wrapper();s=make();s.step(512);saved=s.state_dict()
        ns['_load_checkpoint_megatron']=lambda **kwargs:(101,0)
        ns['load_checkpoint']([],None,s,{},False);self.assertEqual(s.state_dict(),saved)
    def test_hf_path_unmodified(self):
        ns=actual_wrapper();s=make();ns['_is_megatron_checkpoint']=lambda p:False
        ns['_load_checkpoint_hf']=lambda **kwargs: (self.assertIs(kwargs['optimizer'],'optim'),0)[1]
        self.assertEqual(ns['load_checkpoint']([],'optim',s,{},False),0)
    def test_no_object_or_global_class_mutation(self):
        s=make();before=set(vars(s));method=OptimizerParamScheduler.load_state_dict
        native_checkpoint_scheduler(s).load_state_dict(s.state_dict())
        self.assertEqual(set(vars(s)),before);self.assertIs(OptimizerParamScheduler.load_state_dict,method)

if __name__=='__main__':unittest.main(verbosity=2)
