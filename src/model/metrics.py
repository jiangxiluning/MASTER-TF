#   ______                                           __                 
#  /      \                                         /  |                
# /$$$$$$  | __   __   __   ______   _______        $$ |       __    __ 
# $$ |  $$ |/  | /  | /  | /      \ /       \       $$ |      /  |  /  |
# $$ |  $$ |$$ | $$ | $$ |/$$$$$$  |$$$$$$$  |      $$ |      $$ |  $$ |
# $$ |  $$ |$$ | $$ | $$ |$$    $$ |$$ |  $$ |      $$ |      $$ |  $$ |
# $$ \__$$ |$$ \_$$ \_$$ |$$$$$$$$/ $$ |  $$ |      $$ |_____ $$ \__$$ |
# $$    $$/ $$   $$   $$/ $$       |$$ |  $$ |      $$       |$$    $$/ 
#  $$$$$$/   $$$$$/$$$$/   $$$$$$$/ $$/   $$/       $$$$$$$$/  $$$$$$/ 
#
# File: metrics.py
# Author: Owen Lu
# Date: 
# Email: jiangxiluning@gmail.com
# Description:
from typing import *
import tensorflow as tf

class WordAccuary:

    def __init__(self, case_sensitive=False):
        super(WordAccuary, self).__init__()
        self.total = 1e-10
        self.correct = 0
        self.case_sensitive = case_sensitive

    def update(self, y_pred:List[str], y: List[str]):
        self.total += len(y_pred)

        for pred, gt in zip(y_pred, y):
            if not self.case_sensitive:
                pred = pred.lower()
                gt = gt.lower()

            if pred == gt:
                self.correct += 1

    def compute(self):
        return self.correct / self.total

    def reset(self):
        self.total = 1e-10
        self.correct = 0
