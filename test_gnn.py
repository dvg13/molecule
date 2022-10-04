import unittest
import gnn
import tensorflow as tf
import math

generic_message = "Expected result is {}.\n Actual result is {}"

def tensor_approx_equal(a,b,e=.001):
    return tf.math.reduce_max(tf.abs(a - b)) < e
    
class TestGnn(unittest.TestCase):
    def test_padded_mean(self):
        mat_1 = tf.constant([[1,1,1], [0,0,0], [0,0,0]], dtype=tf.float32)
        mat_2 = tf.constant([[1,1,1], [2,2,2], [0,0,0]], dtype=tf.float32)
        mat_3 = tf.constant([[1,1,1], [2,2,2], [3,3,3]], dtype=tf.float32)
        test_matrix = tf.stack([mat_1, mat_2, mat_3])
        
        expected = tf.constant([[1,1,1], [1.5,1.5,1.5], [2,2,2]], dtype=tf.float32)
        actual = gnn.padded_mean(test_matrix)
        
        print(expected)
        print(actual)
        
        self.assertTrue(tensor_approx_equal(expected,actual), generic_message.format(expected,actual))
        
    def test_normalize_adjacency_matrix(self):
        mat = tf.constant([[[1,1,0],[0,1,0],[1,1,1]]], dtype=tf.float32)
        
        expected_standard = tf.constant([[.5,.5,0], [0,1,0], [1/3,1/3,1/3]], dtype=tf.float32)
        actual_standard = gnn.normalize_adjacency_matrix(mat, False, False)
        self.assertTrue(
            tensor_approx_equal(expected_standard,actual_standard),
            generic_message.format(expected_standard,actual_standard)
        )
            
        expected_symmetric = tf.constant(
            [[.5,1/math.sqrt(6),0], 
             [0,1/math.sqrt(3),0], 
             [1/math.sqrt(6),1/3,1/math.sqrt(3)]], 
             dtype=tf.float32
        )
        actual_symmetric = gnn.normalize_adjacency_matrix(mat, True, False)
        self.assertTrue(
            tensor_approx_equal(expected_symmetric,actual_symmetric),
            generic_message.format(expected_symmetric,actual_symmetric)
        )
        
    def test_normalize_adjacency_matrix_results_equal_for_same_degree(self):
        mat = tf.constant([[[1,1,0,0],[1,0,1,0],[0,0,1,1],[0,1,0,1]]], dtype=tf.float32)
        
        actual_standard = gnn.normalize_adjacency_matrix(mat, False, False)
        actual_symmetric = gnn.normalize_adjacency_matrix(mat, True, False)
        
        self.assertTrue(
            tensor_approx_equal(actual_standard,actual_symmetric),
            "Standard result is {}.\n Symmetric result is {}".format(actual_standard,actual_symmetric)
        )
        
if __name__ == '__main__':
    unittest.main()