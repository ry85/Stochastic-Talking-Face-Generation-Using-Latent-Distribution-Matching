class Path(object):
    @staticmethod
    def db_dir(train):
        if train:
            audio_dir = '/dataset_grid/gridcorpus_s3fd/audio/train'
            frame_dir = '/dataset_grid/gridcorpus_s3fd/frame/train'
        else:
            audio_dir = '/dataset_grid/gridcorpus_s3fd/audio/test'
            frame_dir = '/dataset_grid/gridcorpus_s3fd/frame/test'

        return audio_dir, frame_dir
