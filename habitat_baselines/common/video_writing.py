import cv2

from habitat import Config, logger

def get_video_writer(filename="output.avi", fps=20.0, resolution=(640, 480)):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(filename, fourcc, fps, resolution)
    return writer


class VideoWriter:
    # wrapper around video writer that will create video writer and
    # initialize the resolution the first time write happens
    def __init__(self, filename="output.avi", fps=20.0):
        self.filename = filename
        self.fps = fps
        self.writer = None

    def write(self, frame):
        if self.writer is None:
            self.resolution = (frame.shape[1], frame.shape[0])
            self.writer = get_video_writer(
                self.filename, self.fps, self.resolution
            )
        else:
            res = (frame.shape[1], frame.shape[0])
            if res != self.resolution:
                logger.info(
                    f"Warning: video resolution mismatch expected={self.resolution}, frame={res}"
                )
        self.writer.write(frame)

    def release(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
