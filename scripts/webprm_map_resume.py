"""Strict Map prefix resume and cumulative accounting; no implicit approval."""
from __future__ import annotations
import fcntl
import hashlib
import math
import os
from pathlib import Path
import re
import stat
import time
import urllib.error
import urllib.request

PROTOCOL='map_one_additional_transfer_append_prefix_tls3_v1'
SCRATCH_RESERVE=4*2**20
PART_BYTES={'datasets/osm_tile_server.tar':8*2**20,'datasets/nominatim_volumes.tar':512*2**20,
            'datasets/osm_dump.tar':8*2**20,'datasets/osrm_routing.tar':8*2**20}
STAT_FIELDS=('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns','st_mode','st_nlink')


def stat_identity(value):return {key:getattr(value,key) for key in STAT_FIELDS}


def ledger_class(lib,limits,maximum_blob_bytes):
    """An isolated Map ledger; imported charges and reconnects never reset."""
    class Ledger(lib.TransferLedger):
        def __init__(self,root,*,prior_charged_bytes,prior_metadata_bytes,prior_responses):
            counters=(prior_charged_bytes,prior_metadata_bytes,prior_responses)
            if any(type(x) is not int or x<0 for x in counters):raise ValueError('Invalid prior network accounting')
            if (prior_charged_bytes>=limits['download_charged_bytes'] or
                    prior_metadata_bytes>=limits['metadata_received_bytes'] or prior_responses>=limits['max_http_responses']):
                raise ValueError('Original cumulative network caps exhausted')
            self.prior_responses=prior_responses
            super().__init__(root,cap=limits['download_charged_bytes']-prior_charged_bytes,
                             metadata_cap=limits['metadata_received_bytes']-prior_metadata_bytes)
            self.data.update(prior_charged_bytes=prior_charged_bytes,prior_metadata_bytes=prior_metadata_bytes,
                prior_responses=prior_responses,cumulative_cap_bytes=limits['download_charged_bytes'],
                tls_handshake_reconnects=[],maximum_tls_handshake_reconnects=3)
            lib.put(self.path,self.data,replace=True)
        def reserve(self,maximum,url,start,end):
            if len(self.data['reservations'])+self.prior_responses>=limits['max_http_responses']:
                raise ValueError('Original cumulative response cap exhausted')
            return super().reserve(maximum,url,start,end)
        def expand(self,index,maximum):
            row=self.data['reservations'][index];extra=maximum-row['reserved_bytes']
            if (type(maximum) is not int or not 0<maximum<=maximum_blob_bytes or
                    row['state']!='unknown' or row['received_bytes']!=0 or extra<0 or
                    self.data['charged_bytes']+extra>self.cap):
                raise ValueError('Full blob exceeds cumulative Map reservation cap')
            row['reserved_bytes']=maximum;self.data['charged_bytes']+=extra
            lib.put(self.path,self.data,replace=True)
        def retry_tls_handshake(self,index,error):
            reason=getattr(error,'reason',None);row=self.data['reservations'][index]
            if (not isinstance(error,urllib.error.URLError) or not isinstance(reason,TimeoutError) or
                    not re.fullmatch(r'_ssl[.]c:\d+: The handshake operation timed out',str(reason)) or
                    row['state']!='unknown' or row['received_bytes']!=0 or
                    len(self.data['tls_handshake_reconnects'])>=3):return False
            self.data['tls_handshake_reconnects'].append(dict(reservation_id=index,
                reason='pre_response_tls_handshake_timeout',epoch=time.time()))
            lib.put(self.path,self.data,replace=True)
            return True
    return Ledger


class Multipart:
    """Incremental multipart-MD5 using bounded caller buffers, even for512MiB parts."""
    def __init__(self,part_bytes):
        if type(part_bytes) is not int or part_bytes<=0:raise ValueError('Positive multipart boundary required')
        self.part_bytes=part_bytes;self.current=hashlib.md5();self.aggregate=hashlib.md5();self.used=0;self.count=0
    def update(self,data):
        view=memoryview(data)
        while view:
            take=min(len(view),self.part_bytes-self.used);self.current.update(view[:take]);self.used+=take;view=view[take:]
            if self.used==self.part_bytes:
                self.aggregate.update(self.current.digest());self.count+=1;self.current=hashlib.md5();self.used=0
    def etag(self):
        aggregate=self.aggregate.copy();count=self.count
        if self.used:aggregate.update(self.current.digest());count+=1
        return '"'+aggregate.hexdigest()+'-'+str(count)+'"'


def check_prefix(path,proof):
    if (Path(path).resolve()!=Path(path) or any(x.is_symlink() for x in (Path(path),*Path(path).parents)) or
            proof.get('prefix_path')!=str(path) or proof.get('status')!='closed_local_prefix_sha256_verified' or
            proof.get('file_mutated') is not False or proof.get('stat_before')!=proof.get('stat_after') or
            proof.get('fixed_prefix_bytes')!=proof.get('bytes_read') or
            not re.fullmatch('[0-9a-f]{64}',str(proof.get('fixed_prefix_sha256','')))):
        raise ValueError('Canonical verified fixed-prefix proof required')
    current=path.stat()
    if (any(getattr(current,k)!=proof['stat_after'][k] for k in STAT_FIELDS if k!='st_dev') or not stat.S_ISREG(current.st_mode) or
            current.st_nlink!=1 or current.st_size!=proof['fixed_prefix_bytes']):
        raise ValueError('Original partial inode/size/stat changed before append admission')
    return stat_identity(current)


def transfer_dataset(backend,root,item,lib,ledger,deadline,*,prefix=None,part_bytes=None,open_response=None,chunk=64*2**20):
    """Exact206 only. A preserved inode may only grow at its bound prefix end."""
    if item['kind']!='dataset':raise ValueError('Dataset transfer required')
    part_bytes=PART_BYTES[item['filename']] if part_bytes is None else part_bytes
    match=re.fullmatch(r'"([0-9a-f]{32})-(\d+)"',item['etag'])
    if not match or math.ceil(item['bytes']/part_bytes)!=int(match[2]):raise ValueError('Declared multipart boundary/count differs')
    path=backend.checked(Path(prefix['prefix_path']) if prefix else root/item['filename'])
    if prefix:admission_stat=check_prefix(path,prefix)
    else:path.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
    flags=os.O_RDWR|os.O_NOFOLLOW|(os.O_APPEND if prefix else os.O_CREAT|os.O_EXCL)
    fd=os.open(path,flags,0o600);checksum=hashlib.sha256();multipart=Multipart(part_bytes);written=0
    with os.fdopen(fd,'r+b',buffering=0) as output:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if prefix:
            if stat_identity(os.fstat(fd))!=admission_stat:raise ValueError('Partial changed between validation and open')
            while written<prefix['fixed_prefix_bytes']:
                if time.time()>=deadline:raise TimeoutError('Prefix verification deadline exhausted')
                data=output.read(min(2**20,prefix['fixed_prefix_bytes']-written))
                if not data:raise ValueError('Preserved prefix truncated')
                checksum.update(data);multipart.update(data);written+=len(data)
            if checksum.hexdigest()!=prefix['fixed_prefix_sha256'] or stat_identity(os.fstat(fd))!=admission_stat:
                raise ValueError('Preserved prefix SHA/stat changed before first request')
        initial=stat_identity(os.fstat(fd))
        while written<item['bytes']:
            if os.fstat(fd).st_size!=written:raise ValueError('Exclusive append offset changed')
            start=written;end=min(item['bytes']-1,start+chunk-1);maximum=end-start+1
            backend.allowed_delivery(item['url'],item)
            request=urllib.request.Request(item['url'],headers={'Range':f'bytes={start}-{end}',
                'If-Match':item['etag'],'Accept-Encoding':'identity','User-Agent':'OpenWebRL-Map-explicit-prefix-resume'})
            index,response=backend.request_once(lib,ledger,request,maximum,deadline,open_response=open_response)
            received=0;complete=False
            try:
                if hasattr(response.headers,'get_all') and any(len(response.headers.get_all(k,[]))>1 for k in
                        ('Content-Length','Content-Range','ETag','Content-Encoding','Transfer-Encoding')):
                    raise ValueError('Ambiguous dataset response framing')
                if (response.status!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{end}/{item["bytes"]}' or
                        response.headers.get('Content-Length')!=str(maximum) or response.headers.get('ETag')!=item['etag'] or
                        response.headers.get('Content-Encoding','identity')!='identity' or response.headers.get('Transfer-Encoding') is not None):
                    raise ValueError('Exact resumed dataset range/length/ETag identity changed')
                while received<maximum:
                    if time.time()>=deadline:raise TimeoutError('Dataset transfer deadline exhausted')
                    count=min(2**20,maximum-received);data=response.read(count)
                    if not data or len(data)>count:raise ValueError('Truncated or oversized resumed dataset body')
                    received+=len(data)
                    if os.fstat(fd).st_size!=written:raise ValueError('Exclusive append offset changed')
                    view=memoryview(data)
                    while view:
                        count=output.write(view)
                        if not count:raise OSError('Dataset append made no progress')
                        view=view[count:];written+=count
                    checksum.update(data);multipart.update(data)
                output.flush();os.fsync(fd);complete=True
            finally:response.close();ledger.settle(index,received,complete=complete)
        final=os.fstat(fd)
        if (final.st_dev!=initial['st_dev'] or final.st_ino!=initial['st_ino'] or final.st_nlink!=1 or
                final.st_size!=item['bytes'] or multipart.etag()!=item['etag']):
            raise ValueError('Completed dataset inode/length/content multipart ETag mismatch')
        if prefix:
            # Verify the persisted inode after growth, including its original
            # fixed prefix. This extra local read belongs to the same deadline.
            before=stat_identity(final);output.seek(0);readback=hashlib.sha256();prefix_hash=hashlib.sha256()
            readparts=Multipart(part_bytes);seen=0
            while seen<item['bytes']:
                if time.time()>=deadline:raise TimeoutError('Persisted prefix verification deadline exhausted')
                data=output.read(min(2**20,item['bytes']-seen))
                if not data:raise ValueError('Completed dataset truncated during readback')
                if seen<prefix['fixed_prefix_bytes']:prefix_hash.update(data[:prefix['fixed_prefix_bytes']-seen])
                readback.update(data);readparts.update(data);seen+=len(data)
            if (stat_identity(os.fstat(fd))!=before or prefix_hash.hexdigest()!=prefix['fixed_prefix_sha256'] or
                    readback.hexdigest()!=checksum.hexdigest() or readparts.etag()!=item['etag']):
                raise ValueError('Persisted original prefix or completed dataset changed')
        if (path.resolve()!=path or any(x.is_symlink() for x in (path,*path.parents)) or
                stat_identity(path.stat())!=stat_identity(os.fstat(fd))):
            raise ValueError('Completed pathname no longer owns the verified inode')
    return dict(filename=item['filename'],path=str(path),bytes=written,sha256=checksum.hexdigest(),
        published_multipart_etag=item['etag'],computed_multipart_etag=multipart.etag(),multipart_part_bytes=part_bytes,
        multipart_boundary_inference_validated_by_exact_match=True,published_sha256_available=False,
        resumed_prefix_bytes=prefix['fixed_prefix_bytes'] if prefix else 0,
        preserved_prefix_sha256=prefix['fixed_prefix_sha256'] if prefix else None,
        persisted_whole_file_and_prefix_readback_verified=bool(prefix),
        saved_device_id=prefix['stat_after']['st_dev'] if prefix else None,
        worker_device_id=initial['st_dev'],device_id_changed_across_hosts=bool(prefix and initial['st_dev']!=prefix['stat_after']['st_dev']),
        original_inode_preserved=bool(prefix),overwrite_truncate_rename_performed=False)
