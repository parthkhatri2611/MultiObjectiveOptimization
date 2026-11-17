import os
import sys
import types
import unittest
from unittest.mock import patch
from .device_policy import configure_embedding_device


class DevicePolicyTest(unittest.TestCase):
    def test_cpu_does_not_initialize_cuda(self):
        with patch('subprocess.run') as probe:
            self.assertEqual(configure_embedding_device({}), 'cpu')
            probe.assert_not_called()

    def test_no_second_gpu_fails_before_torch_import(self):
        result=types.SimpleNamespace(stdout='0, GPU-generator, Tesla T4\n')
        with patch('subprocess.run',return_value=result):
            with self.assertRaisesRegex(RuntimeError,'Physical GPU 1'):
                configure_embedding_device({'embedding_device':'cuda:1'})

    def test_binds_second_physical_gpu_by_uuid(self):
        result=types.SimpleNamespace(stdout='0, GPU-generator, Tesla T4\n1, GPU-embedding, Tesla T4\n')
        class CUDA:
            @staticmethod
            def is_initialized(): return False
            @staticmethod
            def is_available(): return os.environ.get('CUDA_VISIBLE_DEVICES')=='GPU-embedding'
            @staticmethod
            def device_count(): return 1
            @staticmethod
            def set_device(index): assert index==0
        with patch('subprocess.run',return_value=result), patch.dict(os.environ,{},clear=True), patch.dict(sys.modules,{'torch':types.SimpleNamespace(cuda=CUDA)}):
            self.assertEqual(configure_embedding_device({'embedding_device':'cuda:1'}),'cuda')
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],'GPU-embedding')

    def test_missing_cuda_never_falls_back_to_cpu(self):
        result=types.SimpleNamespace(stdout='1, GPU-embedding, Tesla T4\n')
        cuda=types.SimpleNamespace(is_initialized=lambda:False,is_available=lambda:False)
        with patch('subprocess.run',return_value=result), patch.dict(os.environ,{},clear=True), patch.dict(sys.modules,{'torch':types.SimpleNamespace(cuda=cuda)}):
            with self.assertRaisesRegex(RuntimeError,'not available'):
                configure_embedding_device({'embedding_device':'cuda:1'})
