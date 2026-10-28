"""Offline GitLab reset and evaluator controls; no browser/provider calls."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from openwebrl import webprm_gitlab_native as native
from openwebrl.webprm_cms_hosting import Closed


def row(instance,name,session):
    return dict(status='passed',instance_id=instance,base_sha256='a'*64,user=dict(id=2330,name=name),
                session_sha256=session,native_positive=1,native_negative=0)


class GitLabNativeTests(unittest.TestCase):
    def test_remote_profile_response_commits_before_reload(self):
        events=[];page=SimpleNamespace(saved=False,value='Original')
        response=SimpleNamespace(url=native.ORIGIN+'/-/profile',status=200,request=SimpleNamespace(method='PUT'))
        def finished():events.append('response_finished');page.saved=True;return None
        response.finished=finished
        class Pending:
            value=response
            def __enter__(self):return self
            def __exit__(self,*_):return False
        def expect(predicate,**kwargs):self.assertTrue(predicate(response));return Pending()
        def goto(*args,**kwargs):events.append('reload');page.value='Changed' if page.saved else 'Original'
        page.expect_response=expect;page.wait_for_load_state=lambda *_:None;page.goto=goto
        page.locator=lambda *_:SimpleNamespace(fill=lambda value:events.append('fill'),input_value=lambda:page.value)
        page.get_by_role=lambda *a,**kw:SimpleNamespace(click=lambda:events.append('click'))
        proof={};native.commit_profile_change(page,'Changed',lambda fn,*a,**kw:fn(*a,**kw),proof)
        self.assertLess(events.index('response_finished'),events.index('reload'))
        self.assertEqual(proof['observed_name_after_reload'],'Changed')
        response.status=500
        with self.assertRaises(Closed):native.commit_profile_change(page,'Changed',lambda fn,*a,**kw:fn(*a,**kw),{})

    def test_profile_query_rejects_injection(self):
        self.assertIn(b"username='byteblaze'",native.user_sql('byteblaze'))
        for value in [None,"x' OR true--",'a;DROP','x'*129]:
            with self.subTest(value=value),self.assertRaises(Closed):native.user_sql(value)

    def test_reset_checks_independent_database_session_and_repository(self):
        a,m,b=row('a','Original','s1'),row('a','Changed','s1'),row('b','Original','s2')
        gate=native.reset_gate(a,m,b,repository_marker_absent=True)
        self.assertTrue(gate['fresh_reset_verified']);self.assertFalse(gate['native_git_commit_mutation_tested'])
        bad=[(a,m,dict(b,instance_id='a'),True),(a,m,dict(b,user=m['user']),True),
             (a,m,dict(b,session_sha256='s1'),True),(a,m,b,False),(a,a,b,True),
             (a,m,dict(b,native_negative=1),True),(a,m,dict(b,base_sha256='b'*64),True)]
        for args in bad:
            with self.subTest(args=args),self.assertRaises(Closed):native.reset_gate(*args[:3],repository_marker_absent=args[3])

    def test_evaluator_control_changes_only_expected_text(self):
        calls=[]
        def evaluate(task,trajectory,page,termination):
            calls.append(deepcopy(task));return dict(valid=True,score=1 if len(calls)==1 else 0)
        native.evaluator_controls(SimpleNamespace(evaluate=evaluate),None,dict(id=2330,name='Original'))
        self.assertEqual(calls[0]['eval']['program_html'][0]['locator'],native.NAME_JS)
        self.assertNotEqual(calls[0]['eval']['program_html'][0]['required_contents'],calls[1]['eval']['program_html'][0]['required_contents'])

    def test_user_fixture_requires_one_exact_record(self):
        for value in [[],None,dict(id=True,name='x'),dict(id=0,name='x'),dict(id=1,name=''),dict(id=1,name='x',password='secret')]:
            with self.subTest(value=value),self.assertRaises(Closed):native.validate_user(value)


if __name__=='__main__':unittest.main()
