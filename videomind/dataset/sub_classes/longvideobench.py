# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import nncore
from torch.utils.data import Dataset
import os
import decord
from decord import VideoReader

from videomind.dataset.hybrid import DATASETS
from videomind.utils.parser import parse_query, parse_question


@DATASETS.register(name='longvideobench')
class LongVideoBenchDataset(Dataset):

    ANNO_PATH_VALID = 'data/longvideobench/lvb_val.json'
    ANNO_PATH_TEST = 'data/longvideobench/lvb_test_wo_gt.json'

    VIDEO_ROOT = 'data/longvideobench/videos'

    @classmethod
    def load_annos(self, split='valid'):
        if split == 'valid':
            raw_annos = nncore.load(self.ANNO_PATH_VALID)
        else:
            print('WARNING: Test split does not have ground truth annotations')
            raw_annos = nncore.load(self.ANNO_PATH_TEST)

        annos = []
        for raw_anno in raw_annos:
            vid = raw_anno['video_id']

            if vid.startswith('@'):
                vid = vid[-19:]


            """if not (len(vid) == 11 or len(vid) == 19):
                    print(f"Warning: Invalid video ID length for {vid}, skipping this entry.")
                    continue


            video_path = nncore.join(self.VIDEO_ROOT, vid + '.mp4')
            if not os.path.exists(video_path):
                print(f"Warning: Video file {video_path} not found, skipping this annotation.")
                continue

                # Attempt to load the video with decord
            try:
                vr = VideoReader(video_path)
                # Try to read the first 10 frames (or adjust based on your need)
                frames = vr.get_batch(range(10))
            except decord._ffi.base.DECORDError as e:
                print(f"Warning: Failed to read video {video_path}. Error: {str(e)}")
                # Skip this video if it fails to read
                continue"""

            # videos might come from youtube or other sources
            assert len(vid) in (11, 19)

            anno = dict(
                source='longvideobench',
                data_type='multimodal',
                video_path=nncore.join(self.VIDEO_ROOT, vid + '.mp4'),
                query=parse_query(raw_anno['question']),
                question=parse_question(raw_anno['question']),
                options=raw_anno['candidates'],
                task=str(raw_anno['duration_group']),
                level=raw_anno['level'],
                question_category=raw_anno['question_category'])

            if 'correct_choice' in raw_anno:
                anno['answer'] = raw_anno['candidates'][raw_anno['correct_choice']]
                anno['ans'] = chr(ord('A') + raw_anno['correct_choice'])

            annos.append(anno)

        return annos
