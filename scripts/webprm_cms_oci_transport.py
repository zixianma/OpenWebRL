"""Bounded pinned OCI transport; no source-equivalence or execution claim.

The owner supplies the exact cumulative ledger, including imported charges and
response counts. A registry ignoring Range is accepted only at offset zero,
with an exact complete-blob length reserved durably before any body read.
"""
from __future__ import annotations
import gzip
import hashlib
import os
import time
import urllib.parse
import urllib.request
from scripts import acquire_webprm_cms_original as cms
from scripts import acquire_webprm_map_backend as registry


class Registry(registry.Registry):
    def token(self,item):
        # Reserve sufficient metadata room before the bounded token reader runs.
        # Cached tokens can still be used when the metadata room is exhausted.
        current=self.tokens.get((item['registry'],item['repository']))
        if not (current and time.time()<current[1]) and self.ledger.metadata_cap-self.ledger.data['metadata_received_bytes']<65536:
            raise cms.GateClosed('Cumulative token metadata cap; no request sent')
        return super().token(item)


def transfer_blob(root,item,ledger,deadline,tokens,*,open_response=None,chunk=64*2**20):
    path=cms.checked_path(root/item['filename']);path.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
    checksum=hashlib.sha256();written=0
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as output:
        while written<item['bytes']:
            start=written;end=min(item['bytes']-1,start+chunk-1);url=item['url']
            for redirects in range(3):
                registry.allowed_delivery(url,item)
                headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'}
                # Both pinned Docker Hub and GHCR blobs require origin-scoped
                # authorization. Never forward a registry token to its CDN.
                current=urllib.parse.urlsplit(url);origin=urllib.parse.urlsplit(item['registry'])
                if (current.scheme,current.hostname,current.port or 443)==(origin.scheme,origin.hostname,origin.port or 443):
                    headers['Authorization']='Bearer '+tokens.token(item)
                i,response=registry.request_once(cms,ledger,urllib.request.Request(url,headers=headers),end-start+1,deadline,open_response=open_response)
                received=0;complete=False
                try:
                    if hasattr(response.headers,'get_all') and any(len(response.headers.get_all(k,[]))>1 for k in
                        ('Content-Length','Content-Range','Content-Encoding','Transfer-Encoding','Location')):
                        raise cms.GateClosed('Ambiguous blob response framing')
                    if response.status in (301,302,303,307,308):
                        if redirects==2:raise cms.GateClosed('Blob redirect bound exhausted')
                        location=response.headers.get('Location','')
                        if not location:raise cms.GateClosed('Blob redirect omitted delivery URL')
                        complete=response.headers.get('Content-Length')=='0'
                        url=registry.allowed_delivery(urllib.parse.urljoin(url,location),item);continue
                    full=response.status==200 and start==0 and response.headers.get('Content-Length')==str(item['bytes'])
                    if full:
                        if response.headers.get('Content-Range') is not None:raise cms.GateClosed('Contradictory full blob framing')
                        ledger.expand(i,item['bytes']);end=item['bytes']-1
                    expected=end-start+1
                    if (not full and (response.status!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{end}/{item["bytes"]}') or
                            response.headers.get('Content-Length')!=str(expected) or
                            response.headers.get('Content-Encoding','identity')!='identity' or response.headers.get('Transfer-Encoding') is not None):
                        raise cms.GateClosed('Pinned blob exact range/length contract unavailable')
                    while received<expected:
                        if time.time()>=deadline:raise cms.GateClosed('Blob transfer deadline exhausted')
                        maximum=min(2**20,expected-received);data=response.read(maximum)
                        if not data or len(data)>maximum:raise cms.GateClosed('Truncated or oversized blob response')
                        output.write(data);checksum.update(data);received+=len(data);written+=len(data)
                    complete=True;break
                finally:response.close();ledger.settle(i,received,complete=complete)
            else:raise cms.GateClosed('No accepted blob response')
        output.flush();os.fsync(output.fileno())
    if checksum.hexdigest()!=item['sha256']:raise cms.GateClosed('Pinned compressed blob SHA256 differs')
    return dict(filename=item['filename'],bytes=written,sha256=checksum.hexdigest())


def verify_raw_layers(root,items,deadline,*,maximum=12*2**30):
    total=0;proof=[]
    for item in items:
        checksum=hashlib.sha256();count=0
        with gzip.open(cms.checked_path(root/item['filename']),'rb') as stream:
            while True:
                if time.time()>=deadline:raise cms.GateClosed('Raw layer verification deadline exhausted')
                data=stream.read(min(2**20,maximum-total+1))
                if not data:break
                total+=len(data);count+=len(data)
                if total>maximum:raise cms.GateClosed('Expanded raw layer byte cap exceeded')
                checksum.update(data)
        if 'sha256:'+checksum.hexdigest()!=item['raw_diff_id']:raise cms.GateClosed('Pinned decompressed raw diff ID differs')
        proof.append(dict(compressed_sha256=item['sha256'],raw_diff_id=item['raw_diff_id'],raw_bytes=count))
    return dict(unique_layers=proof,total_raw_bytes=total,all_raw_diff_ids_verified=True,extraction_performed=False)


