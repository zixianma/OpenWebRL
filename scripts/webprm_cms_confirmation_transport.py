"""Bounded original-CMS Drive confirmation; prospective use requires new approval.

No network occurs on import. The exact original file ID, fixed TLS form action,
single confirmation, complete response framing, and cumulative caps are mandatory.
"""
from __future__ import annotations
import email.message
import hashlib
from html.parser import HTMLParser
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from scripts import acquire_webprm_cms_original as cms

GateClosed=cms.GateClosed
LIMITS=cms.LIMITS
SOURCE_URL=cms.SOURCE_URL
DRIVE_ID=cms.DRIVE_ID
ARCHIVE_NAME=cms.ARCHIVE_NAME
ARCHIVE_BYTES=cms.ARCHIVE_BYTES
CHUNK=cms.CHUNK
BLOCK=cms.BLOCK
METADATA_RESPONSE_CAP=cms.METADATA_RESPONSE_CAP
checked_path=cms.checked_path
allowed_url=cms.allowed_url
opener=cms.opener
put=cms.put

class Confirmation(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.forms=[];self.fields={};self.inside=False;self.closed=False
    def handle_starttag(self, tag, attrs):
        pairs=list(attrs);a=dict(pairs)
        if len(a)!=len(pairs):raise GateClosed('Ambiguous confirmation attributes')
        if tag=='form':
            if self.forms or self.inside:raise GateClosed('Exactly one confirmation form required')
            if a!={'id':'download-form','action':'https://drive.usercontent.google.com/download','method':'get'}:
                raise GateClosed('Confirmation form identity/action/method differs')
            self.forms.append(a);self.inside=True
        elif tag=='input' and self.inside:
            if a.get('type')=='submit' and not a.get('name'):return
            if (a.get('type')!='hidden' or set(a)-{'type','name','value'} or
                    a.get('name') not in ('id','export','confirm','uuid') or a['name'] in self.fields):
                raise GateClosed('Unexpected or duplicate confirmation input')
            self.fields[a['name']]=a.get('value','')
    def handle_endtag(self,tag):
        if tag=='form':
            if not self.inside:raise GateClosed('Unmatched confirmation form')
            self.inside=False;self.closed=True

def confirmation_url(raw):
    if not isinstance(raw,bytes) or len(raw)>METADATA_RESPONSE_CAP:
        raise GateClosed('Confirmation metadata exceeds cap')
    try:
        parser=Confirmation();parser.feed(raw.decode('utf-8'));parser.close()
    except (UnicodeError,ValueError) as error:
        raise GateClosed('Invalid bounded confirmation form') from error
    fields=parser.fields
    if (not parser.closed or parser.inside or len(parser.forms)!=1 or set(fields)!={'id','export','confirm','uuid'} or
            fields['id']!=DRIVE_ID or fields['export']!='download' or fields['confirm']!='t' or
            re.fullmatch('[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',fields['uuid']) is None):
        raise GateClosed('Original Drive file/confirmation identity differs')
    return 'https://drive.usercontent.google.com/download?'+urllib.parse.urlencode(fields)

class CumulativeLedger(cms.TransferLedger):
    def __init__(self, root, *, prior_charged_bytes=0, prior_metadata_bytes=0, prior_responses=0,
                 cap=10000000000, metadata_cap=16*2**20):
        if any(type(x) is not int or x<0 for x in (prior_charged_bytes,prior_metadata_bytes,prior_responses)):
            raise GateClosed('Explicit prior network accounting required')
        self.prior_responses=prior_responses
        super().__init__(root,cap=cap-prior_charged_bytes,metadata_cap=metadata_cap-prior_metadata_bytes)
        self.data.update(prior_charged_bytes=prior_charged_bytes,prior_metadata_bytes=prior_metadata_bytes,
                         prior_responses=prior_responses,cumulative_cap_bytes=cap)
        put(self.path,self.data,replace=True)
    def reserve(self, maximum,url,start,end):
        if len(self.data['reservations'])+self.prior_responses>=LIMITS['max_http_responses']:
            raise GateClosed('Cumulative HTTP response cap')
        return super().reserve(maximum,url,start,end)
    def expand(self,index,maximum):
        row=self.data['reservations'][index]
        extra=max(0,maximum-row['reserved_bytes'])
        if (row['state']!='unknown' or row['received_bytes']!=0 or type(maximum) is not int or
                maximum<=0 or maximum>ARCHIVE_BYTES or self.data['charged_bytes']+extra>self.cap):
            raise GateClosed('Full-body reservation would exceed cumulative cap')
        row['reserved_bytes']+=extra;self.data['charged_bytes']+=extra
        put(self.path,self.data,replace=True)

def download(root, ledger, deadline, *, open_response=None, size=ARCHIVE_BYTES, chunk=CHUNK):
    """One contiguous range transfer; no retry/resume or unbounded response read."""
    open_response = opener() if open_response is None else open_response
    target = checked_path(root/'archive-attempt-001.partial')
    url, etag, offset = SOURCE_URL, None, 0
    confirmation_used = False
    checksum = hashlib.sha256()
    with target.open('xb') as output:
        os.chmod(target, 0o600)
        while offset < size:
            end = min(offset+chunk,size)-1
            for redirect in range(LIMITS['max_redirects_per_range']+1):
                if time.time() >= deadline:
                    raise GateClosed('Transfer work deadline reached')
                allowed_url(url)
                maximum = max(end-offset+1, METADATA_RESPONSE_CAP)
                index = ledger.reserve(maximum, url, offset, end)
                received, complete, metadata = 0, False, False
                response = None
                try:
                    headers = {'Range':f'bytes={offset}-{end}','Accept-Encoding':'identity'}
                    if etag is not None:
                        headers['If-Match'] = etag
                    request = urllib.request.Request(url, headers=headers)
                    try:
                        response = open_response(request, timeout=max(.1,min(10,deadline-time.time())))
                    except urllib.error.HTTPError as error:
                        response = error
                    status = response.status
                    if hasattr(response.headers,'get_all') and any(
                            len(response.headers.get_all(key,[]))>1 for key in
                            ('Content-Length','Content-Range','ETag','Content-Disposition',
                             'Content-Encoding','Transfer-Encoding','Location')):
                        raise GateClosed('Ambiguous duplicate response identity or framing headers')
                    length = response.headers.get('Content-Length','')
                    length = int(length) if length.isdigit() else None
                    full_body = status == 200 and length == size and offset == 0
                    if status != 206 and not full_body:
                        metadata = True
                        budget = min(METADATA_RESPONSE_CAP, ledger.metadata_cap-ledger.data['metadata_received_bytes'])
                        if budget <= 0:
                            raise GateClosed('Metadata body cap exhausted')
                        count = min(budget,length) if length is not None else budget
                        raw = response.read(count)
                        if len(raw)>count:
                            raise GateClosed('Transport exceeded requested metadata read bound')
                        received = len(raw)
                        complete = length is not None and len(raw) == length
                        fd = os.open(root/f'response-{index:04d}-metadata.bin', os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW, 0o600)
                        with os.fdopen(fd,'wb') as evidence:
                            evidence.write(raw); evidence.flush(); os.fsync(evidence.fileno())
                        if status in (301,302,303,307,308) and complete:
                            if redirect == LIMITS['max_redirects_per_range']:
                                raise GateClosed('Official redirect bound exhausted')
                            location = response.headers.get('Location','')
                            if not location:
                                raise GateClosed('Official redirect omitted its location')
                            url = allowed_url(urllib.parse.urljoin(url,location))
                            continue
                        if status == 200 and complete and offset == 0 and not confirmation_used:
                            confirmed = confirmation_url(raw)
                            if redirect == LIMITS['max_redirects_per_range']:
                                raise GateClosed('Confirmation exceeds reviewed redirect/metadata step bound')
                            put(root/f'confirmation-{index:04d}.json',dict(
                                protocol='official_drive_fixed_cms_confirmation_v1',
                                metadata_sha256=hashlib.sha256(raw).hexdigest(),
                                request_url=confirmed, original_drive_id=DRIVE_ID,
                                at_archive_offset=offset, one_confirmation_only=True))
                            url, confirmation_used = confirmed, True
                            continue
                        raise GateClosed('Source quota/HTML/unavailable or unbounded metadata; no retry or mirror fallback')
                    if full_body:
                        if response.headers.get('Content-Range') is not None:
                            raise GateClosed('Full archive response has contradictory partial-range framing')
                        ledger.expand(index, size)
                        end = size-1
                    if (length != end-offset+1 or not full_body and response.headers.get('Content-Range') != f'bytes {offset}-{end}/{size}' or
                            response.headers.get('Content-Encoding','identity') != 'identity' or
                            response.headers.get('Transfer-Encoding') is not None):
                        raise GateClosed('Exact original file range/length/encoding contract unavailable')
                    current_etag = response.headers.get('ETag','')
                    header = email.message.Message(); header['content-disposition'] = response.headers.get('Content-Disposition','')
                    if (not re.fullmatch(r'"[^"\r\n]+"',current_etag) or
                            etag is not None and current_etag != etag or header.get_filename() != ARCHIVE_NAME):
                        raise GateClosed('Strong consistent entity or exact original filename unavailable')
                    etag = current_etag
                    while received < length:
                        if time.time() >= deadline:
                            raise GateClosed('Transfer work deadline reached')
                        count = min(BLOCK,length-received)
                        block = response.read(count)
                        if len(block)>count:
                            raise GateClosed('Transport exceeded requested archive read bound')
                        if not block:
                            raise GateClosed('Partial response body; no automatic retry')
                        received += len(block); output.write(block); checksum.update(block)
                    output.flush(); os.fsync(output.fileno())
                    complete = True
                    offset = end+1
                    break
                finally:
                    if response is not None:
                        response.close()
                    ledger.settle(index, received, complete=complete, metadata=metadata)
            else:
                raise GateClosed('Redirect loop')
    final = checked_path(root/ARCHIVE_NAME)
    if final.exists() or final.is_symlink():
        raise GateClosed('Original completed archive path already exists')
    os.rename(target,final)
    result = dict(status='complete_official_location_transfer', archive_path=str(final), bytes=offset,
        archive_sha256=checksum.hexdigest(), strong_etag=etag, official_drive_id=DRIVE_ID,
        final_url_sha256=hashlib.sha256(url.encode()).hexdigest(), final_host=urllib.parse.urlsplit(url).hostname,
        historical_checksum_independently_published=False)
    put(root/'download-complete.json',result)
    return result

