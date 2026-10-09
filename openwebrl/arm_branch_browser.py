"""Private, opt-in browser evidence for the continuation-branch experiment."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

STATE_JS = r"""() => {
 const rect = e => { const r=e.getBoundingClientRect(); return [r.x,r.y,r.width,r.height].map(x=>Math.round(x*10)/10); };
 const describe = e => ({tag:e.tagName,id:e.id,name:e.getAttribute('name'),type:e.getAttribute('type'),
   value:e.value ?? null,checked:e.checked ?? null,selectedIndex:e.selectedIndex ?? null,
   disabled:!!e.disabled,rect:rect(e)});
 let session; try {session=Object.fromEntries(Object.keys(sessionStorage).sort().map(k=>[k,sessionStorage[k]]));}
 catch(e){session={unavailable:e.name};}
 return {url:location.href,title:document.title,text:document.body?.innerText||'',
   scroll:[scrollX,scrollY],history_length:history.length,session,
   forms:Array.from(document.querySelectorAll('input,textarea,select')).map(describe),
   focus:document.activeElement ? describe(document.activeElement) : null};
}"""


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def visible_state(snapshot):
    """Drop session evidence only; preserve page/forms/focus/history exactly."""
    value = deepcopy(snapshot)
    value.pop('storage', None)
    for page in value.get('pages', []):
        page.pop('session', None)
    return value


class BrowserTopologyChanged(RuntimeError):
    """The tab list or active page changed while evidence was being read."""


async def browser_state(env):
    pages = list(env.context.pages)
    active_page = env.page
    if active_page not in pages:
        raise BrowserTopologyChanged("Active page is absent from the captured tab list")
    state = [await page.evaluate(STATE_JS) for page in pages]
    storage = await env.context.storage_state(indexed_db=True)
    if env.page is not active_page or list(env.context.pages) != pages:
        raise BrowserTopologyChanged("Browser tabs changed while reading branch state")
    # Absolute cookie expiry is tied to collection wall time. Preserve it in the
    # raw receipt but compare values, scope and flags; expired cookies disappear.
    compare = deepcopy(storage)
    for cookie in compare.get("cookies", []):
        cookie.pop("expires", None)
    compare["cookies"] = sorted(compare.get("cookies", []), key=canonical)
    compare["origins"] = sorted(compare.get("origins", []), key=canonical)
    return dict(pages=state, active_tab=pages.index(active_page), storage=compare), storage


def branch_environment_class(parent, mismatch_error, protocol='observable-replay-v1'):
    class BranchEnvironment(parent):
        async def _capture_once(self):
            # Bracket a single capture, inside the parent's bounded retry loop.
            # Spanning that whole loop would reject a successfully retried frame
            # merely because hydration changed the discarded first frame.
            try:
                before, _ = await browser_state(self)
                observation = await super()._capture_once()
                after, storage = await browser_state(self)
            except BrowserTopologyChanged as exc:
                self.controlled_journal("branch_capture_rejected", dict(reason=str(exc)))
                # Retry observation only, under the unchanged parent's three
                # attempts / 15-second deadline. Never choose a substitute tab.
                raise mismatch_error(str(exc)) from exc
            project = visible_state if protocol == 'visible-replay-v2' else lambda x: x
            if project(before) != project(after):
                self.controlled_journal("branch_capture_rejected", dict(
                    before=before, after=after,
                    screenshot_sha256=observation['controlled']['screenshot_sha256']))
                raise mismatch_error("Branch state changed during observation")
            observation["branch_snapshot"] = after
            observation["branch_snapshot_sha256"] = digest(after)
            observation['branch_replay_protocol'] = protocol
            self.controlled_journal("branch_state", dict(snapshot=after,
                storage_with_expiry=storage, digest=digest(after)))
            return observation
    return BranchEnvironment


def install():
    from openwebrl import controlled_browser_server as server
    original = server.controlled_environment

    def factory(base, manifest):
        protocol = manifest.get('branch_state_version')
        if protocol not in ('observable-replay-v1', 'visible-replay-v2'):
            raise ValueError("Explicit branch browser protocol required")
        return branch_environment_class(original(base, manifest), server.ObservationMismatch, protocol)

    server.controlled_environment = factory
    server.main()


if __name__ == "__main__":
    install()
