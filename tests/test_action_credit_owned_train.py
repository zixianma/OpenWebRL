import ast, pathlib, unittest
SOURCE=pathlib.Path(__file__).resolve().parents[1]/'openwebrl/action_credit_bootstrap.py'
module=ast.parse(SOURCE.read_text())
selected=ast.Module(body=[n for n in module.body if isinstance(n,ast.ClassDef) and n.name in ('_OwnedBuilder','_OwnedRay','_OwnedTrainRay')],type_ignores=[])
def require(v,msg):
    if not v:raise ValueError(msg)
ns={'require':require};exec(compile(selected,str(SOURCE),'exec'),ns)
class Base:pass
class Wrong:pass
class Owner:
    def __init__(self):self.actors=[]
class Remote:
    def __init__(self,ray,cls,kw):self.ray,self.cls,self.kw=ray,cls,kw
    def options(self,**kw):self.ray.options.append(kw);return self
    def remote(self,*a,**kw):
        if self.ray.fail:raise RuntimeError('creation failed')
        x=object();self.ray.created.append((x,a,kw));return x
class FakeRay:
    def __init__(self):self.declarations=[];self.options=[];self.created=[];self.fail=False
    def remote(self,**kw):
        def decorate(cls):self.declarations.append((cls,kw));return Remote(self,cls,kw)
        return decorate
    def kill(self,actor,**kw):return 'killed'
def factory(base):
    class RPC(base):pass
    return RPC
class Tests(unittest.TestCase):
    def setUp(self):
        self.ray=FakeRay();self.owner=Owner();self.adapter=ns['_OwnedTrainRay'](self.ray,self.owner,Base,factory)
    def test_decorator_exact_base_and_owner_before_return(self):
        builder=self.adapter.remote(num_gpus=1,runtime_env={'env_vars':{'X':'1'}})(Base)
        handle=builder.options(num_cpus=2).remote('arg',named='value')
        self.assertIs(self.owner.actors[0][1],handle);self.assertEqual(self.owner.actors[0][0],'train')
        self.assertTrue(issubclass(self.ray.declarations[0][0],Base));self.assertIsNot(self.ray.declarations[0][0],Base)
        self.assertEqual(self.ray.options,[{'num_cpus':2}]);self.assertEqual(self.ray.created[0][1:],(('arg',),{'named':'value'}))
    def test_direct_class_form_preserves_options(self):
        self.adapter.remote(Base,num_gpus=1).remote();self.assertEqual(self.ray.declarations[0][1],{'num_gpus':1})
    def test_wrong_class_refused_before_ray(self):
        for form in (lambda:self.adapter.remote(num_gpus=1)(Wrong),lambda:self.adapter.remote(Wrong,num_gpus=1)):
            with self.assertRaises(ValueError):form()
        self.assertFalse(self.ray.declarations)
    def test_multiple_classes_refused_before_ray(self):
        with self.assertRaises(ValueError):self.adapter.remote(Base,Wrong)
        self.assertFalse(self.ray.declarations)
    def test_failed_creation_not_recorded_as_owned(self):
        self.ray.fail=True
        with self.assertRaises(RuntimeError):self.adapter.remote(Base).remote()
        self.assertFalse(self.owner.actors)
    def test_bad_subclass_refused(self):
        with self.assertRaises(ValueError):ns['_OwnedTrainRay'](self.ray,self.owner,Base,lambda _:Wrong)
    def test_kill_removes_only_exact_handle(self):
        a=self.adapter.remote(Base).remote();b=self.adapter.remote(Base).remote();self.adapter.kill(a,no_restart=True)
        self.assertEqual(self.owner.actors,[('train',b)])
    def test_transport_discards_owner_and_substitution(self):
        restore,args=self.adapter.__reduce__();self.assertEqual(restore.__module__,'importlib');self.assertEqual(args,('ray',))
    def test_production_leaves_canonical_symbol_unmodified(self):
        train=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='train')
        calls=[n for n in ast.walk(train) if isinstance(n,ast.Call)]
        self.assertTrue(any(isinstance(n.func,ast.Name) and n.func.id=='_OwnedTrainRay' for n in calls))
        self.assertFalse(any(isinstance(n.func,ast.Name) and n.func.id=='_patched' and len(n.args)>1 and isinstance(n.args[1],ast.Constant) and n.args[1].value=='MegatronTrainRayActor' for n in calls))
if __name__=='__main__':unittest.main(verbosity=2)
