import hashlib
import importlib.util
import io
from pathlib import Path
import tarfile
import tempfile
import time
import unittest

spec=importlib.util.spec_from_file_location('map_host',Path(__file__).resolve().parents[1]/'scripts/prepare_webprm_map_hosting.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class MapHostingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=m.RUNTIME/'next-stage-preparation',prefix='map-host-test-');self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)

    def archive(self,items,gzip=False):
        path=self.root/('fixture.tar.gz' if gzip else 'fixture.tar')
        with tarfile.open(path,'w:gz' if gzip else 'w') as tar:
            for name,content in items:
                item=tarfile.TarInfo(name);item.size=len(content);tar.addfile(item,io.BytesIO(content))
        return path,hashlib.sha256(path.read_bytes()).hexdigest()

    def test_bounded_gzip_inventory_finds_frontend_database_evidence(self):
        path,h=self.archive([('openstreetmap-website/config/settings.yml',b'original'),('openstreetmap-website/db/dump.sql',b'fixture')],True)
        proof=m.inspect(path,h,'frontend_source',time.time()+5)
        self.assertFalse(proof['extracted']);self.assertFalse(proof['ready_to_host'])
        self.assertEqual(len(proof['frontend_database_candidates']),2)

    def test_volume_layout_follows_observed_five_component_prefix(self):
        items=[('mnt/projects/ogma3/docker/volumes/'+role+'/_data/file',b'base') for role in ('osm-data','osm-tiles')]
        path,h=self.archive(items);proof=m.inspect(path,h,'osm_tile_server.tar',time.time()+5)
        self.assertEqual(set(proof['role_totals']),{'osm-data','osm-tiles'})
        with self.assertRaises(m.Closed):m.dataset_destination('osm_tile_server.tar','projects/ogma3/docker/volumes/osm-data/_data/file')

    def test_missing_transport_profile_is_rejected(self):
        path,h=self.archive([('car/us-northeast-latest.osrm',b'car'),('bike/us-northeast-latest.osrm',b'bike')])
        with self.assertRaises(m.Closed):m.inspect(path,h,'osrm_routing.tar',time.time()+5)

    def test_path_traversal_duplicate_and_excess_expansion_are_rejected(self):
        for name in ('/etc/passwd','../escape','a/../../escape','a\\escape'):
            with self.assertRaises(m.Closed):m.clean_name(name)
        path,h=self.archive([('db/a',b'12345')])
        with self.assertRaises(m.Closed):m.inspect(path,h,'frontend_source',time.time()+5,expanded_byte_cap=4)

    def test_wrong_acquisition_hash_and_expired_budget_rejected(self):
        path,h=self.archive([('manifest.json',b'[]')])
        with self.assertRaises(m.Closed):m.inspect(path,'0'*64,'docker_save',time.time()+5)
        with self.assertRaises(m.Closed):m.inspect(path,h,'docker_save',time.time()-1)

    def source(self):
        source=self.root/'source';source.mkdir()
        values={'config/settings.yml':b'nominatim_url: original\nfossgis_osrm_url: original\n',
          'vendor/assets/leaflet/leaflet.osm.js':(m.RUNTIME/'next-stage-preparation/map-gasse-template-leaflet.osm.js').read_bytes(),
          'app/assets/javascripts/index/directions/fossgis_osrm.js':(m.RUNTIME/'next-stage-preparation/map-gasse-template-fossgis_osrm.js').read_bytes()}
        for name,raw in values.items():
            path=source/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
        return source,values

    def test_actual_pinned_templates_route_three_modes_and_preserve_base(self):
        source,values=self.source();overlay=self.root/'overlay';proof=m.configure_frontend(source,overlay)
        for name,raw in values.items():self.assertEqual((source/name).read_bytes(),raw)
        route=(overlay/'app/assets/javascripts/index/directions/fossgis_osrm.js').read_text()
        for port in (':5000',':5001',':5002'):self.assertIn(port,route)
        self.assertNotIn('__OSMCarSuffix__',route);self.assertFalse(proof['host_services_started'])

    def test_configuration_drift_fails_before_any_overlay_write(self):
        source,_=self.source();(source/'config/settings.yml').write_text('wrong: drift\n')
        overlay=self.root/'overlay'
        with self.assertRaises(m.Closed):m.configure_frontend(source,overlay)
        self.assertFalse(overlay.exists())

    def test_frontend_symlink_and_protected_home_rejected(self):
        with self.assertRaises(m.Closed):m.checked('/gpfs/home/zixianma/never-write')
        link=self.root/'link';link.symlink_to(self.root)
        with self.assertRaises(m.Closed):m.checked(link)

    def test_private_backend_port_collisions_are_explicitly_bridged(self):
        spec=m.service_spec();mapping={x['service']:(x['internal_port'],x['cell_loopback_port']) for x in spec['bridges']}
        self.assertEqual(mapping['tile'],(80,8080));self.assertEqual(mapping['nominatim'],(8080,8085))
        self.assertEqual(mapping['osrm-bike'],(5000,5001));self.assertFalse(spec['ready_to_host'])


if __name__=='__main__':unittest.main()
