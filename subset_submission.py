# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 03:48:00 2017

@author: yuegong3
"""

import numpy as np

prediction = np.round(np.load('result.npy'))



tag_name = ['faces', 'left_foot', 'visual_digits', 'left_hand', 'calculation',
            'language', 'horizontal_checkerboard', 'human_sound',
            'vertical_checkerboard', 'objects', 'places', 'scramble',
            'right_hand', 'right_foot', 'visual_words', 'visual',
            'non_human_sound', 'auditory', 'saccades']



with open('subset_submission.csv', 'w') as f:
    print("id,tags", file=f)
    for i in range(1971):
        label = []
        for j in range(19):
            if prediction[i, j]:
                label.append(tag_name[j])
        print("%d,%s" % (i, ' '.join(label)), file=f)


label_prediction = np.load('label_result.npy')
with open('label_subset_submission.csv', 'w') as f:
    print("id,tags", file=f)
    for i in range(1971):
        label = []
        for j in range(19):
            if label_prediction[i, j]:
                label.append(tag_name[j])
        print("%d,%s" % (i, ' '.join(label)), file=f)