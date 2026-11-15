"""Regression checks for numeric parsing and explicit inference flags."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from ocr_learned import predict

class StubModel:
    def predict_proba(self,texts):
        return np.array([[.01,.99] for _ in texts])

class ParsingTests(unittest.TestCase):
    def setUp(self):self.bundle={'model':StubModel(),'catalog':[]}
    def parse(self,text):
        with patch('ocr_learned.line_groups',return_value=[{'text':text,'median_y':0}]):
            return predict({},100,self.bundle)['items'][0]
    def test_dosage_not_counted_as_quantity(self):
        item=self.parse('Лекарство 0.30мг*15мг 1480.00')
        self.assertEqual(item['quantity'],'1')
        self.assertTrue(item['quantity_inferred'])
        self.assertEqual(item['sum'],'1480.00')
        self.assertTrue(item['sum_inferred'])
    def test_quantity_and_spaced_price(self):
        item=self.parse('Молоко 199. 99*2 399.98')
        self.assertEqual((item['quantity'],item['price'],item['sum']),('2','199.99','399.98'))
        self.assertFalse(item['quantity_inferred'])
        self.assertFalse(item['sum_inferred'])
if __name__=='__main__':unittest.main()
