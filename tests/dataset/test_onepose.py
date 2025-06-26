import unittest
import os
from omegaconf import OmegaConf
from src.datasets.onepose import OnePoseDataset
import numpy as np
class TestOnePoseDataset(unittest.TestCase):
    '''
         # below are the dataset configs
    OnePose:
      name: 'OnePose'
      config:
        base: 
          image_size : ${image_size}
          length: 6 # persequence length
          dynamic_length: False # if true, each batch will have different length, and length should be a list(min,max)
          stride: [5, 25]
          dynamic_stride: True # if true, each batch will have different stride, and stride should be a list(min,max)
          random_stride: False # if true, dynamic_stride will be ignored
          augmentation_method: null # will be supported in the future
          pose_augmentation: False
          intri_augmentation: False
          compute_optical: False
          max_norm: False # or avg norm
          precision: ${precision}
        
        root: "${hydra:runtime.cwd}/data/onepose"
  
    '''
    def setUp(self):
        # Create a temporary config for testing
        self.config = OmegaConf.create({
            'base': {
                'image_size': 224,
                'length': 6,
                'dynamic_length': False,
                'stride': [5, 25],
                'dynamic_stride': True,
                'random_stride': False,
                'augmentation_method': None,
                'pose_augmentation': False,
                'intri_augmentation': False,
                'compute_optical': False,
                'max_norm': False,
                'precision': 'float32'
            },
            'root': os.path.join(os.getcwd(), 'data/onepose')
        })
        self.split = 'train'
        self.dataset = OnePoseDataset(self.config, self.split)
        
        

    def test_initialization(self):
        self.assertEqual(self.dataset.dataset, 'onepose')
        self.assertEqual(self.dataset.split, self.split)
        self.assertEqual(self.dataset.root, self.config.root)


    def test_load_data(self):
        self.dataset.load_data()
        self.assertIn('train', self.dataset.images)
        self.assertIn('ref', self.dataset.images)
        self.assertIn('train', self.dataset.boxes)
        self.assertIn('ref', self.dataset.boxes)
        self.assertIn('train', self.dataset.poses)
        self.assertIn('ref', self.dataset.poses)
        self.assertIn('train', self.dataset.intrinsics)
        self.assertIn('ref', self.dataset.intrinsics)
        self.assertIn('train', self.dataset.cat_len)
        self.assertIn('ref', self.dataset.cat_len)

    def test_load_data_from_dir(self):
        self.dataset.load_data('test')
        self.assertIn('test', self.dataset.images)
        self.assertIn('test', self.dataset.boxes)
        self.assertIn('test', self.dataset.poses)
        self.assertIn('test', self.dataset.intrinsics)
        self.assertIn('test', self.dataset.cat_len)
        
    def test_all_file_sorted(self):
        self.dataset.load_data()
        for key in self.dataset.images['train']:
            self.assertEqual(self.dataset.images['train'][key], sorted(self.dataset.images['train'][key]))
        for key in self.dataset.boxes['train']:
            self.assertEqual(self.dataset.boxes['train'][key], sorted(self.dataset.boxes['train'][key]))
        for key in self.dataset.poses['train']:
            self.assertEqual(self.dataset.poses['train'][key], sorted(self.dataset.poses['train'][key]))
        for key in self.dataset.intrinsics['train']:
            self.assertEqual(self.dataset.intrinsics['train'][key], sorted(self.dataset.intrinsics['train'][key]))
        for key in self.dataset.cat_len['train']:
            self.assertEqual(self.dataset.cat_len['train'][key], len(self.dataset.images['train'][key]))
            
    def all_file_have_same_len(self):
        self.dataset.load_data()
        for key in self.dataset.images['train']:
            self.assertEqual(len(self.dataset.images['train'][key]), len(self.dataset.boxes['train'][key]))
            self.assertEqual(len(self.dataset.images['train'][key]), len(self.dataset.poses['train'][key]))
            self.assertEqual(len(self.dataset.images['train'][key]), len(self.dataset.intrinsics['train'][key]))
            self.assertEqual(len(self.dataset.images['train'][key]), self.dataset.cat_len['train'][key])

if __name__ == '__main__':
    unittest.main()