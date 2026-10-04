from pathlib import Path
import tempfile
import unittest

from openwebrl.env.browser_runtime import browser_process_environment


class BrowserRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='arm-browser-env-test-')
        self.addCleanup(self.directory.cleanup)
        self.mesa = Path(self.directory.name) / '50_mesa.json'
        self.mesa.write_text('{}')

    def test_software_browser_isolation_preserves_actor_environment(self):
        actor = {'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7', 'PATH': '/usr/bin'}
        child = browser_process_environment(['--disable-gpu'], environ=actor, mesa_vendor=self.mesa)
        self.assertEqual(child['__EGL_VENDOR_LIBRARY_FILENAMES'], str(self.mesa))
        self.assertEqual(child['CUDA_VISIBLE_DEVICES'], actor['CUDA_VISIBLE_DEVICES'])
        self.assertEqual(actor, {'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7', 'PATH': '/usr/bin'})

    def test_explicit_vendor_and_hardware_browser_are_preserved(self):
        explicit = {'__EGL_VENDOR_LIBRARY_FILENAMES': '/custom/vendor.json'}
        self.assertEqual(browser_process_environment(['--disable-gpu'], environ=explicit,
            mesa_vendor=self.mesa), explicit)
        self.assertEqual(browser_process_environment(['--enable-gpu'], environ={},
            mesa_vendor=self.mesa), {})

    def test_missing_mesa_keeps_original_environment(self):
        self.assertEqual(browser_process_environment(['--disable-gpu'], environ={'PATH': '/usr/bin'},
            mesa_vendor=self.mesa.with_name('missing.json')), {'PATH': '/usr/bin'})


if __name__ == '__main__':
    unittest.main()
