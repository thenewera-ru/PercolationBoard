import matplotlib.pyplot as plt
import matplotlib as mtp
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

from abc import ABC, abstractmethod
from typing import Callable, Tuple, Optional


class IPlot(ABC):
    '''
    Interface for convenient plotting using "Seaborn" and "Matplotlib".
    This is just an API for convenience.
    '''

    class Private:
        
        def __init__(self, data=None):
            self.data = data

    params = {
        'figure.figsize': (11.7, 8.27),
        'font.size': 7,
        'axes.titlesize': 1,
        'axes.labelsize': 12.5,
        'font.family': 'serif',
    }

    palettes = {
        "blue": sns.color_palette("GnBu_d"),
        "green": sns.cubehelix_palette(8, start=2, rot=0, dark=0, light=.95, reverse=True),
        "purple": sns.dark_palette("purple"),
        "navy": sns.light_palette("navy", reverse=True)
    }

    def __init__(self, nrows=1, ncols=1, data=None, params=None):
        '''
        :param nrows: -
        :param ncols: -
        :param data: [optional]
        :param params: -
        '''
        assert nrows >= 1 and ncols >= 1
        assert params is not None

        self._ = Plot.Private(data)
        self._.palette = self.palettes['blue']
        self.params = params

        self.title = None
        self.x_as = 'X'
        self.y_as = 'Y'

        self.nrows = nrows
        self.ncols = ncols

    @property
    def data(self):
        return self._.data

    @data.setter
    def data(self, value):
        self._.data = value

    @property
    def pallete(self):
        return self._.pallete
    
    @data.setter
    def pallete(self, value):
        self._.pallete = value

    def title_as(self, title_name):
        self.title_as = title_name
        return self

    def x_as(self, x_name):
        self.x_as = xTitle
        return self

    def y_as(self, yTitle):
        self.y_as = yTitle
        return self

    @abstractmethod
    def compile(self, it, xs, **params):
        '''
        :return: Self object. For chaining method calls...
        '''
        pass

    @abstractmethod
    def show(self):
        '''
        :return: Self object. For chaining method calls...
        '''
        pass

    @abstractmethod
    def save(self, filepath, dpi=250, extension='pdf'):
        '''
        :return: Self object. For chaining method calls...
        '''
        pass

    @abstractmethod
    def close(self):
        '''
        :return: Nothing. End of chain call
        '''
        pass

    def __add__(self, other):
        pass

    def __repr__(self):
        return self.data



class Plot(IPlot):

    def __init__(self, nrows=1, ncols=1, data=None, style='white', params=IPlot.params):
        super(Plot, self).__init__(data=data, params=params)
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "sans-serif"
        })
        sns.set_style(style)
        self.canvas, self.axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=self.params['figure.figsize'], sharex=False, sharey=False)
        self.axes = np.atleast_2d(self.axes).T
        self.rotate_as = None

    def rotate_as(self, angle):
        self.rotate_as = angle
        return self

    def compile(self, it, xs, **params):
        assert len(it) == len(xs)
        data = pd.DataFrame(columns=['x'])
        n = params.get('n', 1000)
        for i, iCall in enumerate(it):
            s, e = xs[i]
            step = (e - s) / n
            xxs = np.array([0.0] * n)
            yys = np.array([0.0] * n)
            for j in range(n):
                xxs[j] = (s + step * j)
                yys[j] = iCall(xxs[j])
            di = pd.DataFrame({'x': xxs, str(i): yys})
            data = pd.merge(data, di, on='x', how='outer')
        subscriptions = params.get('subscriptions', [str(i) for i in list(np.arange(len(it)))])
        assert len(subscriptions) == len(xs)
        data.columns = ['x'] + subscriptions
        self._.data = pd.melt(data, id_vars=['x'], var_name='-', value_name='F = F(x)')
        return self

    def rotate(self, p):
        if self.rotate_as:
            for tick in p.get_xticklabels():
                tick.set_rotation(self.rotation)
        return p

    def view(self, p):
        bot, top = p.get_ylim()
        height = top - bot
        p.margins(y=0.1, x=0.1)
        return p

    def subscription(self, p):
        '''
        :param p: sns.barplot(...options)
        :return: sns.barplot(...options)
        '''
        if self.title is not None:
            p.set_title(self.title, fontsize=self.params['axes.titlesize'])
        if self.x_as is not None:
            p.set_xlabel(self.x_as, fontsize=self.params['font.size'])
        if self.y_as is not None:
            p.set_ylabel(self.y_as, fontsize=self.params['font.size'])
        return p

    def show(self, row=0, col=0):
        '''
        :return:
        '''
        if self._.data is None:
            raise ValueError('Plot is not compiled before plotting (calling plot.show())')
        sns.set(
            rc=self.params,
            palette=sns.light_palette("navy", reverse=True)
        )
        p = sns.lineplot(x='x', y='F = F(x)', data=self._.data, hue='-', ax=self.axes[row][col])
        p = self.rotate(p)
        p = self.view(p)
        p = self.subscription(p)
        plt.show(block=False)
        self.axes[row][col] = p
        return self

    def save(self, filepath, filename='graph', dpi=250, extension='pdf'):
        Path(filepath).mkdir(parents=True, exist_ok=True)
        self.canvas.savefig(str(Path(filepath) / filename) + '.' + extension, bbox_inches='tight', dpi=dpi)
        return self

    def close(self):
        mtp.rc_file_defaults()
        plt.close(self.canvas)

    def __repr__(self):
        return 'Plot({!r})'.format(self.data)


def save_as_pdf(f: Callable, range: Tuple, filepath: str, dpi: Optional[int] = 250):
    p = Plot(style='white')
    # p = p.palette_as(p.palettes['navy'])
    p = p.compile([f], xs=[range], subscriptions=['Y = Y(x)'])
    p = p.show()
    p = p.save(filepath=filepath, dpi=250)
    p = p.close()