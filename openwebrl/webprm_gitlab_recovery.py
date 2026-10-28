"""Read-only verification of an unexecuted, preserved OCI base for recovery."""
import hashlib
import os
from pathlib import Path, PurePosixPath
import stat

from .webprm_cms_materializer import Closed, check, name, private_path


def verify_pristine_base(base, manifest, deadline):
    base=private_path(base)
    if not base.is_dir():raise Closed('Preserved pristine base root is absent or not a directory')
    if (manifest.get('destination')!=str(base) or manifest.get('image_executed') is not False
            or manifest.get('source')!='third_party_digest_pinned_oci_candidate'):
        raise Closed('Exact unexecuted original materialization required')
    entries=manifest['entries'];expected=set();hashes=manifest['regular_file_sha256']
    for path,row in entries.items():
        if name(path)!=path:raise Closed('Canonical inventory path required')
        if row['kind']!='device':expected.add(path)
        expected.update(str(p) for p in PurePosixPath(path).parents if str(p)!='.')
    observed=set();apparent=0
    def unreadable(error):raise Closed('Preserved pristine base inventory unreadable: '+str(error))
    for directory,dirs,files in os.walk(base,followlinks=False,onerror=unreadable):
        for leaf in dirs+files:
            check(deadline);path=Path(directory)/leaf;relative=str(path.relative_to(base));observed.add(relative)
            st=path.lstat();apparent+=st.st_size
            row=entries.get(relative,dict(kind='dir',mode=0o755));kind=row['kind']
            mode=stat.S_IMODE(st.st_mode)
            if kind=='dir':
                if not stat.S_ISDIR(st.st_mode) or mode!=row['mode']:raise Closed('Preserved base directory changed')
            elif kind=='symlink':
                if not stat.S_ISLNK(st.st_mode) or os.readlink(path)!=row['linkname']:raise Closed('Preserved base symlink changed')
            elif kind=='fifo':
                if not stat.S_ISFIFO(st.st_mode) or mode!=row['mode']:raise Closed('Preserved base FIFO changed')
            elif kind in ('file','hardlink'):
                target=entries[row['regular_target']] if kind=='hardlink' else row
                if not stat.S_ISREG(st.st_mode) or mode!=target['mode'] or st.st_size!=row['bytes']:
                    raise Closed('Preserved base regular metadata changed')
                if kind=='hardlink':
                    other=(base/row['regular_target']).lstat()
                    if (st.st_dev,st.st_ino)!=(other.st_dev,other.st_ino):raise Closed('Preserved base hardlink identity changed')
                else:
                    digest=hashlib.sha256();fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
                    with os.fdopen(fd,'rb') as stream:
                        opened=os.fstat(stream.fileno())
                        while True:
                            check(deadline);chunk=stream.read(2**20)
                            if not chunk:break
                            digest.update(chunk)
                        after=os.fstat(stream.fileno())
                    identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
                    if identity(opened)!=identity(after) or identity(st)!=identity(after) or digest.hexdigest()!=hashes[relative]:
                        raise Closed('Preserved pristine base file content changed')
            else:raise Closed('Unexpected preserved base member')
    if observed!=expected:
        raise Closed('Preserved pristine base inventory differs: observed='+str(len(observed))+
            ',expected='+str(len(expected))+',missing='+repr(sorted(expected-observed)[:5])+
            ',extra='+repr(sorted(observed-expected)[:5]))
    return dict(verified=True,base=str(base),apparent_bytes=apparent,entries=len(observed),
                full_regular_hash_readback=True,image_executed=False,modified=False)


def verify_independent_copy(base, state, manifest, deadline, other_state=None):
    base,state=private_path(base),private_path(state)
    if base==state or base in state.parents or state in base.parents:raise Closed('Disjoint full-copy state required')
    other_inodes=set()
    if other_state is not None:
        other_state=private_path(other_state)
        if other_state==state:raise Closed('Two distinct fresh states required')
        for directory,dirs,files in os.walk(other_state,followlinks=False):
            for leaf in dirs+files:
                check(deadline);st=(Path(directory)/leaf).lstat();other_inodes.add((st.st_dev,st.st_ino))
    expected=dict(manifest['entries'])
    for path in list(expected):
        for parent in PurePosixPath(path).parents:
            if str(parent)!='.':expected.setdefault(str(parent),dict(kind='dir'))
    checked=directories=0
    for path,row in expected.items():
        check(deadline)
        if row['kind']=='device':continue
        original=(base/path).lstat();copied=(state/path).lstat()
        if stat.S_IFMT(original.st_mode)!=stat.S_IFMT(copied.st_mode):raise Closed('Private state changed original member type')
        inode=(copied.st_dev,copied.st_ino)
        if (original.st_dev,original.st_ino)==inode or inode in other_inodes:
            raise Closed('Private state shares an inode with pristine base or previous state')
        if row['kind'] in ('file','hardlink'):checked+=1
        if row['kind']=='dir':directories+=1
    return dict(verified=True,regular_inodes_checked=checked,directory_inodes_checked=directories,
                shared_regular_inodes=0,other_state_checked=other_state is not None)
