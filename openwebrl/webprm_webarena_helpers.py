"""Explicit adapters for the three helpers used by released single-site tasks.

Native function/query bodies are pinned. Only their lazy deployment imports are
removed in favor of explicit owned-site configuration. Native ``eval`` calls
dispatch an exact task-derived expression; they never execute Python strings.
This module establishes offline contracts, not website/reset readiness.
"""
from __future__ import annotations

import __future__
import ast
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
import hashlib
import ipaddress
import json
from pathlib import Path
import re
from types import SimpleNamespace
from urllib.parse import urlparse, urlsplit, urlencode
import urllib.request

HELPER_FILE = 'viewer/src/evaluation_harness/helper_functions.py'
HELPER_SHA256 = '499907cc05c864d163b9520bb0c086a71ffc6a44e403e67e75b51ab104150080'
REDDIT = 'reddit_get_post_url'
GITLAB = 'gitlab_get_project_memeber_role'  # Released spelling is intentional.
SHOPPING = 'shopping_get_latest_order_url'


class HelperNotReady(ValueError):
    pass


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise HelperNotReady('Owned Shopping helper does not follow redirects')


class OwnedShoppingRequests:
    """Two bounded local REST calls per evaluator, with no proxy or retries."""

    def __init__(self, origin):
        parsed = urlsplit(origin)
        try:
            loopback = ipaddress.ip_address(parsed.hostname).is_loopback
            port = parsed.port
        except (ValueError, TypeError):
            loopback, port = False, None
        if (parsed.scheme != 'http' or not loopback or port is None or
                parsed.path not in ('', '/') or parsed.username or parsed.password or
                parsed.query or parsed.fragment):
            raise HelperNotReady('Explicit loopback HTTP Shopping origin and port required')
        self.origin = origin.rstrip('/')
        self.calls = ContextVar('shopping_helper_calls', default=None)

    @contextmanager
    def session(self):
        token = self.calls.set(0)
        try:
            yield
        finally:
            self.calls.reset(token)

    def request(self, method, url, *, headers, data=None, params=None):
        expected = {'POST': '/rest/default/V1/integration/admin/token',
                    'GET': '/rest/V1/orders'}
        count = self.calls.get()
        if (count is None or count >= 2 or method not in expected or
                url != self.origin + expected[method]):
            raise HelperNotReady('Shopping helper request outside explicit session/endpoint cap')
        self.calls.set(count + 1)
        if params:
            url += '?' + urlencode(params)
        payload = data.encode() if data is not None else None
        request = urllib.request.Request(url, data=payload, headers=headers, method=method)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
        with opener.open(request, timeout=10) as response:
            if response.status != 200:
                raise HelperNotReady('Shopping helper HTTP request failed')
            raw = response.read(2**20 + 1)
            if len(raw) > 2**20:
                raise HelperNotReady('Shopping helper response exceeded one MiB')
            result = json.loads(raw)
        return SimpleNamespace(status_code=200, json=lambda: result)

    def post(self, url, **kwargs):
        return self.request('POST', url, **kwargs)

    def get(self, url, **kwargs):
        return self.request('GET', url, **kwargs)


def _parse(expression, field):
    try:
        node = ast.parse(expression.removeprefix('func:'), mode='eval').body
    except (ValueError, SyntaxError) as error:
        raise HelperNotReady('Malformed native helper expression') from error
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.keywords:
        raise HelperNotReady('Unsupported native helper expression')
    name, args = node.func.id, node.args
    if name == REDDIT and field == 'url' and len(args) == 1:
        if isinstance(args[0], ast.Constant) and args[0].value == '__last_url__':
            return name, ()
    elif name == SHOPPING and field == 'url' and not args:
        return name, ()
    elif name == GITLAB and field == 'locator' and len(args) == 2:
        if (isinstance(args[0], ast.Name) and args[0].id == '__page__' and
                isinstance(args[1], ast.Constant) and isinstance(args[1].value, str) and
                re.fullmatch(r'[A-Za-z0-9_.-]+', args[1].value)):
            return name, (args[1].value,)
    raise HelperNotReady('Native helper call is outside the released single-site contract')


class NativeHelpers:
    def __init__(self, root, *, shopping_origin=None, shopping_accounts=None):
        path = Path(root) / HELPER_FILE
        source = path.read_bytes()
        if hashlib.sha256(source).hexdigest() != HELPER_SHA256:
            raise HelperNotReady('Pinned native helper source changed')
        self.requests = None
        names = {REDDIT, GITLAB}
        namespace = dict(json=json, urlparse=urlparse)
        if shopping_origin is not None or shopping_accounts is not None:
            self.requests = OwnedShoppingRequests(shopping_origin)
            try:
                account = shopping_accounts['shopping_site_admin']
                if any(not isinstance(account[k], str) or not account[k] for k in ('username', 'password')):
                    raise ValueError('Missing benchmark account')
            except (KeyError, TypeError, ValueError) as error:
                raise HelperNotReady('Explicit Shopping benchmark admin account required') from error
            namespace.update(SHOPPING=self.requests.origin, ACCOUNTS=deepcopy(shopping_accounts),
                             requests=self.requests)
            names.update({SHOPPING, 'shopping_get_auth_token'})
        tree = ast.parse(source)
        definitions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        if {n.name for n in definitions} != names:
            raise HelperNotReady('Missing pinned helper definitions')

        class DeploymentImports(ast.NodeTransformer):
            def visit_ImportFrom(self, node):
                if node.module != 'webarbiter.src.browser_env.env_config':
                    raise HelperNotReady('Unexpected native helper import')
                return None

        module = ast.fix_missing_locations(DeploymentImports().visit(
            ast.Module(body=definitions, type_ignores=[])))
        exec(compile(module, str(path), 'exec', flags=__future__.annotations.compiler_flag), namespace)
        self.functions = {name: namespace[name] for name in names if name != 'shopping_get_auth_token'}
        self.bindings = ContextVar('native_helper_bindings', default=None)
        self.manifest = dict(protocol='explicit_native_single_site_helpers_v1',
            source_sha256=HELPER_SHA256, functions=sorted(self.functions),
            deployment_imports='explicit_injection', expression_execution='exact_dispatch',
            shopping_origin=self.requests.origin if self.requests else None,
            website_readiness_verified=False)

    def validate_task(self, task):
        needed = set()
        for target in task['eval'].get('program_html', []):
            for field in ('url', 'locator'):
                expression = target.get(field, '')
                if expression.startswith('func:'):
                    name, _ = _parse(expression, field)
                    if name not in self.functions:
                        raise HelperNotReady('Helper needs explicitly configured owned deployment')
                    if name == GITLAB:
                        needed.add('evaluate')
        return needed

    @contextmanager
    def bind(self, task, page):
        self.validate_task(task)
        bindings = []
        for target in task['eval'].get('program_html', []):
            for field in ('url', 'locator'):
                expression = target.get(field, '')
                if not expression.startswith('func:'):
                    continue
                name, args = _parse(expression, field)
                bindings.append((expression[5:], name, args, page))
        token = self.bindings.set(bindings)
        try:
            if self.requests:
                with self.requests.session():
                    yield
            else:
                yield
        finally:
            self.bindings.reset(token)

    def evaluate_expression(self, expression):
        bindings = self.bindings.get()
        for template, name, args, page in bindings or []:
            expected = template
            if name == REDDIT:
                # Match native text substitution without executing Python. Use
                # the current page because earlier HTML targets can navigate.
                terminal_url = page.url
                expected = template.replace('__last_url__', terminal_url)
                args = (terminal_url,)
            elif name == GITLAB:
                expected = template.replace('__page__', 'page')
                args = (page, *args)
            if expression == expected:
                return self.functions[name](*args)
        raise HelperNotReady('Unbound or changed native helper expression')
