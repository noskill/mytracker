import os
import cv2
import pandas as pd
from pandas.core.indexes.api import ensure_index_from_sequences
from utils import visualization as vis
import utils.datasets as datasets
import numpy

MAX_FRAME = 10000000

names = ['FrameId', 'Id', 'X', 'Y', 'Width', 'Height', 'Confidence', 'ClassId', 'Visibility']

COL, ROW, WIDTH, HEIGHT, CONFIDENCE, CLASSID, VISIBILITY = 0, 1, 2, 3, 4, 5, 6

# https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = numpy.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = numpy.ones(window_len, 'd')
    else:
        w = eval('numpy.' + window + '(window_len)')

    y = numpy.convolve(w / w.sum(), s, mode='valid')
    y = y[1:-1]
    y[0] = x[0]
    y[-1] = x[-1]
    return y


class Editor:
    def __init__(self, dataloader, df, save_path):
        cv2.namedWindow('frame', cv2.WINDOW_GUI_EXPANDED | cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('frame', self.on_mouse, 0)
        self.frame = None
        self.img = None
        self.frame_id = 0
        self.dataloader = dataloader
        self.df = df
        switch = '0 : OFF \n1 : ON'
        nameb1 = "narrower"
        nameb2 = "wider"
        nameb3 = 'longer'
        nameb4 = 'shorter'
        namebdel = 'delete edited'
        namebdel_track = 'delete track'
        change_id_from_current = 'change id from current'
        exchange_id_from_current = 'exchange tracks from current'
        save = 'save'

        name_interpolate = 'interpolate'
        name_to_table = 'commit edited'
        smooth_name = 'smooth'

        cv2.createButton(nameb1, lambda *args: self.handle_button(nameb1, *args))
        cv2.createButton(nameb2, lambda *args: self.handle_button(nameb2, *args))
        cv2.createButton(nameb3, lambda *args: self.handle_button(nameb3, *args))
        cv2.createButton(nameb4, lambda *args: self.handle_button(nameb4, *args))
        cv2.createButton(namebdel, lambda *args: self.handle_button(namebdel, *args))

        cv2.createButton(name_interpolate, lambda *args: self.handle_button(name_interpolate, *args))
        cv2.createButton(namebdel_track, lambda *args: self.handle_button(namebdel_track, *args))
        cv2.createButton(change_id_from_current, lambda *args: self.handle_button(change_id_from_current, *args))
        cv2.createButton(exchange_id_from_current, lambda *args: self.handle_button(exchange_id_from_current, *args))
        cv2.createButton(save, lambda *args: self.handle_button(save, *args))
        cv2.createButton(name_to_table, lambda *args: self.handle_button(name_to_table, *args))
        cv2.createButton(smooth_name, lambda *args: self.handle_button(smooth_name, *args))
        self.editor_frame = None
        self.editor_track = dict()
        self.save_path = save_path

    def save_csv(self):
        self.df = self.df.sort_index()
        with open(self.save_path, 'wt') as f:
            f.write(self.df.to_csv(index=True, header=False))

    def track_to_table(self):
        new_id = int(input("track id:"))
        col_dict = dict()
        keys = sorted(self.editor_track.keys())
        col_dict['FrameId'] = numpy.asarray(keys)
        col_dict['Id'] = numpy.stack([new_id for k in keys]).squeeze().astype(numpy.int16)
        tlwh = numpy.stack([self.editor_track[k][0] for k in keys]).squeeze()
        col_dict['X'] = tlwh[:, COL]
        col_dict['Y'] = tlwh[:, ROW]
        col_dict['Width'] = tlwh[:, WIDTH]
        col_dict['Height'] = tlwh[:, HEIGHT]
        col_dict['Confidence'] = tlwh[:, CONFIDENCE]
        col_dict['ClassId'] = tlwh[:, CLASSID]
        col_dict['Visibility'] = tlwh[:, VISIBILITY]
        arrays = col_dict['FrameId'], col_dict['Id']
        index = ensure_index_from_sequences(arrays, names=('FrameId', 'Id'))
        dfnew = pd.DataFrame(col_dict, columns=[x for x in names if x not in index.names], index=index)
        self.df = self.df.append(dfnew)
        self.df = self.df.sort_index()
        self.editor_track.clear()

    def change_track_id(self, old_id, new_track_id, start_frame_id=0, end_frame_id=MAX_FRAME):
        old_data = self.df.xs(old_id, level='Id')
        old_data = old_data[old_data.index >= start_frame_id]
        old_data = old_data[old_data.index <= end_frame_id]
        col_dict = {'FrameId': old_data.index.values,
                    'Id': (new_track_id,) * len(old_data),
                    }
        for name in old_data.columns:
            col_dict[name] = old_data[name].values
        arrays = col_dict['FrameId'], col_dict['Id']
        index = ensure_index_from_sequences(arrays, names=('FrameId', 'Id'))
        dfnew = pd.DataFrame(col_dict, columns=[x for x in names if x not in index.names], index=index)
        self.delete_from_table(old_id, start_frame_id, end_frame_id)
        self.df = self.df.append(dfnew)
        self.df = self.df.sort_index()


    def delete_from_table(self, track_id, frame=0, end_frame=MAX_FRAME):
        frames = self.df.xs(track_id, level='Id').index.values
        frames = frames[frames >= frame]
        frames = frames[frames <= end_frame]
        keys = [(frame, track_id) for frame in frames]
        print('deleting:')
        print(keys)
        self.df = self.df.drop(keys)

    def handle_button(self, button_name, *args):
        if button_name == "narrower":
            self.narrower()
        elif button_name == 'wider':
            self.wider()
        elif button_name == 'longer':
            self.higher()
        elif button_name == 'shorter':
            self.lower()
        elif button_name == 'delete track':
            self.delete_track()
        elif button_name == 'delete edited':
            self.delete_edited()
        elif button_name == 'interpolate':
            self.interpolate()
        elif button_name == 'change id from current':
            self.change_track_id_from_current()
        elif button_name == 'exchange tracks from current':
            self.exchange_from_current()
        elif button_name == 'save':
            self.save_csv()
        elif button_name == 'commit edited':
            self.track_to_table()
        elif button_name == 'smooth':
            self.smooth()
        self.redraw_track()

    def smooth(self):
        keys = sorted(self.editor_track.keys())
        stacked = numpy.stack([self.editor_track[k][0] for k in keys]).squeeze()
        smoothed = numpy.stack([smooth(stacked[:, i], window_len=7, window='flat') for i in range(stacked.shape[1])]).T
        for i, key in enumerate(keys):
            self.editor_track[key][0] = smoothed[i]

    def change_track_id_from_current(self):
        source = int(input("old track id:"))
        target = int(input("new track id:"))
        until = int(input('for number of frames:'))
        print("renaming {0} to {1}".format(source, target))
        self.change_track_id(source, target, self.frame_id, self.frame_id + until)

    def exchange_from_current(self):
        source = int(input("old track id:"))
        target = int(input("new track id:"))
        until = int(input('for number of frames:'))
        print("swapping {0} and {1}".format(source, target))
        self.change_track_id(source, 9888, self.frame_id, self.frame_id + until)
        self.change_track_id(target, 9889, self.frame_id, self.frame_id + until)
        self.change_track_id(9888, target, self.frame_id, self.frame_id + until)
        self.change_track_id(9889, source, self.frame_id, self.frame_id + until)

    def delete_track(self):
        to_delete = int(input('track to delete:'))
        start_from = self.frame_id
        until = int(input('for number of frames:'))
        self.delete_from_table(to_delete, start_from, start_from + until)

    def interpolate(self):
        source_frame = sorted(list(self.editor_track.keys()))[-2]
        source_tlwh = self.editor_track[source_frame][0]
        target_tlwh = self.editor_track[self.frame_id][0]
        results = []
        new_frames = list(range(source_frame + 1, self.frame_id))
        for i in range(len(source_tlwh)):
            results.append(numpy.interp(new_frames,
                                        [source_frame, self.frame_id],
                                        [source_tlwh[i], target_tlwh[i]]))
        for frame_id in new_frames:
            i = frame_id - source_frame - 1
            tlwh = numpy.asarray([results[x][i] for x in range(len(source_tlwh))])
            self.editor_track[frame_id] = [tlwh, self.editor_track[source_frame][1]]

    def delete_edited(self):
        if self.frame_id in self.editor_track:
            self.editor_track.pop(self.frame_id)

    def on_mouse(self, event, x, y, flags, params, boxes=[]):
        if event == cv2.EVENT_LBUTTONDOWN:
            print('Start Mouse Position: ' + str(x) + ', ' + str(y))
            sbox = [x, y]
            boxes.append(sbox)
            # print count
            # print sbox

        elif event == cv2.EVENT_LBUTTONUP:
            print('End Mouse Position: ' + str(x) + ', ' + str(y))
            ebox = [x, y]
            boxes.append(ebox)
        if len(boxes) >= 2:
            height = max(x[1] for x in boxes) - min(x[1] for x in boxes)
            width = max(x[0] for x in boxes) - min(x[0] for x in boxes)
            top_left = min(x[1] for x in boxes), min(x[0] for x in boxes)
            print(top_left, height, width)
            self.process_box(*top_left, height, width)
            boxes.clear()

    def next(self):
        i = 0
        while True:
            i = max(i, 0)
            self.frame_id = i + 1
            self.frame_id = min(self.frame_id, len(self.dataloader) - 1)
            i = self.frame_id - 1
            _, _, img = self.dataloader[self.frame_id]
            self.img = img
            # img_path = os.path.join(img_dir, f)
            # img = cv2.imread(img_path)
            #cv2.imshow("img", img)

            frame_data = None
            try:
                frame_data = self.df.xs(self.frame_id, level='FrameId')
            except KeyError as e:
                self.frame = img
            if frame_data is not None:
                tlwh = frame_data[['X', 'Y', 'Width', 'Height']].to_numpy()
                track_id = frame_data.index.values
                self.frame = vis.plot_tracking(img,
                                          tlwh,
                                          track_id,
                                          frame_id=self.frame_id,
                                          fps=1.)

            self.show(self.frame)
            self.redraw_track()
            # cv2.imwrite(os.path.join('./results', '{:05d}.png'.format(self.frame_id)), self.frame)
            # i += 1

            if i >= 6599:
                break
            k = cv2.waitKey(0)

            if k == 8:  # back
                i -= 10
            if k == 113:  # q
                break
            elif k == 13:  # enter
                i += 10
            elif k == 32:  # space
                i += 1
            elif k == 119:  # w
                self.up()
            elif k == 115:  # s
                self.down()
            elif k == 97:  # a
                self.left()
            elif k == 100:  # d
                self.right()

    def up(self):
        value = -1
        modify = ROW
        self.update_track(modify, value)

    def down(self):
        value = 1
        modify = ROW
        self.update_track(modify, value)

    def left(self):
        value = -1
        modify = COL
        self.update_track(modify, value)

    def right(self):
        value = 1
        modify = COL
        self.update_track(modify, value)

    def narrower(self):
        value = -1
        modify = WIDTH
        self.update_track(modify, value)

    def wider(self):
        value = 1
        modify = WIDTH
        self.update_track(modify, value)

    def lower(self):
        value = -1
        modify = HEIGHT
        self.update_track(modify, value)

    def higher(self):
        value = 1
        modify = HEIGHT
        self.update_track(modify, value)

    def update_track(self, modify, value):
        if self.editor_track and self.frame_id in self.editor_track:
            self.editor_track[self.frame_id][0][modify] += value
            for i in range(1, 24):
                frame_id = self.frame_id + i
                if frame_id in self.editor_track:
                    self.editor_track[frame_id][0][modify] += value * (0.9 ** i)
                frame_id = self.frame_id - i
                if frame_id in self.editor_track:
                    self.editor_track[frame_id][0][modify] += value * (0.9 ** i)

    def redraw_track(self):
        if self.editor_track and self.frame_id in self.editor_track:
            self.editor_frame = vis.plot_tracking(self.frame, [self.editor_track[self.frame_id][0][:4]],
                                                  self.editor_track[self.frame_id][1],
                                                  frame_id=self.frame_id, fps=1.)
            self.show(self.editor_frame)

    def process_box(self, row, col, height, width):
        tlwh = numpy.asarray([col, row, width, height, 1.0, -1, 1.0])
        track_id = numpy.ones(1) * 1000
        self.editor_track[self.frame_id] = [tlwh,
                              track_id]
        if self.frame is not None:
            self.redraw_track()

    def show(self, frame):
        #frame = cv2.resize(self.frame, (2*self.dataloader.width, 2*self.dataloader.height), interpolation=cv2.INTER_LANCZOS4)
        cv2.imshow('frame', frame)



names = ['FrameId', 'Id', 'X', 'Y', 'Width', 'Height', 'Confidence', 'ClassId', 'Visibility']


def show_box():
    video_path = '/mnt/fileserver/shared/datasets/cameras/Odessa/Duke_on_the_left/fragments/child/child_set005_00:16:20-00:20:00.mp4'
    dataloader = datasets.LoadVideo(video_path, img_size = (1920, 1080))
    path = '/mnt/fileserver/shared/datasets/MOT/MOT17Det/train/MOT17-04/'
    fname = os.path.join(path, 'gt', 'gt.txt')
    img_dir = os.path.join(path, 'img1')
    files = os.listdir(img_dir)
    sep = r'\s+|\t+|,'
    min_confidence = 0.5
    fname = '/home/noskill/projects/Towards-Realtime-MOT/result.csv'
    #fname = '/mnt/fileserver/shared/datasets/MOT/MOT17Det/train/MOT17-04/gt/gt.txt'
    df = pd.read_csv(
        fname,
        sep=',',
        index_col=[0, 1],
        skipinitialspace=True,
        header=None,
        names=names,
        engine='python'
    )

    # Remove all rows without sufficient confidence
    df = df[df['Confidence'] >= min_confidence]
    editor = Editor(dataloader, df, fname)
    editor.next()


show_box()
