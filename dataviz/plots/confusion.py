import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_curve, auc, f1_score,
                             precision_recall_curve, average_precision_score)

#from .utils import Colors
#from tqdm import tqdm

from bokeh.plotting import figure
from bokeh.layouts import widgetbox, row, column, layout
from bokeh.models import ColumnDataSource, HoverTool, Div
from bokeh.models.widgets import Slider
from bokeh.io import curdoc


class ROCData(object):

    separations = range(1,100,1)
    proportions = range(1,100,1)


    def __init__(self):
        self.generate_data()


    def generate_data(self):
        self.left = np.random.random(size=100)*10.
        self.right = np.random.random(size=100)*10.

        self.left_accel = (np.random.random(size=100)*0.3)+0.7
        self.right_accel = (np.random.random(size=100)*0.3)+0.7


    def separate_vectors(self, left, right, left_accel, right_accel, separation):

        right_diffs = 10. - right
        left_diffs = left

        left_diffs = left_accel * (left_diffs * separation)
        right_diffs = right_accel * (right_diffs * separation)

        return (left - left_diffs), (right + right_diffs)


    def find_max_separation(self, left_separations, right_separations):
        max_separation = 0.
        for s in self.separations:
            min_left = np.min(left_separations)
            max_right = np.max(right_separations)
            if max_right < min_left:
                max_separation = s
            else:
                break
        return max_separation


    def get_data_balance(self, p=50):
        left = self.left[:p]
        right = self.right[:(100-p)]
        left_accel = self.left_accel[:p]
        right_accel = self.right_accel[:(100-p)]
        return left, right, left_accel, right_accel


    def fit_logreg(self, left_sep, right_sep):

        y = np.concatenate([np.ones(len(right_sep)), np.zeros(len(left_sep))])
        X = np.concatenate([right_sep, left_sep])[:,np.newaxis]

        lr = LogisticRegression()
        lr.fit(X, y)
        y_pp = lr.predict_proba(X)[:, 1]
        y_pred = lr.predict(X)

        fpr_, tpr_, _ = roc_curve(y, y_pp)
        auc_ = auc(fpr_, tpr_)
        acc_ = np.abs(0.5 - np.mean(y)) + 0.5

        precision, recall, _ = precision_recall_curve(y, y_pp)
        avg_precision = average_precision_score(y, y_pp)
        f1 = f1_score(y, y_pred)

        return dict(mod=lr,
                    fpr=fpr_,
                    tpr=tpr_,
                    auc=auc_,
                    acc=acc_,
                    precision=precision,
                    recall=recall,
                    avg_precision=avg_precision,
                    f1=f1,
                    y_pred=y_pred,
                    y_pp=y_pp,
                    y_true=y,
                    X=X.ravel(),
                    ones=right_sep,
                    zeros=left_sep)


    def calculate(self, proportion, separation):
        left, right, left_accel, right_accel = self.get_data_balance(p=proportion)
        left_sep, right_sep = self.separate_vectors(left, right,
                                                    left_accel, right_accel,
                                                    separation/100.)

        model_info = self.fit_logreg(left_sep, right_sep)
        return model_info


    def precalculate_all_data(self):
        self.class_balances = {}

        for p in self.proportions:
            self.class_balances[p] = {}
            for s in self.separations:
                model_info = self.calculate(p, s)
                self.class_balances[p][s] = model_info



def calculate_threshold(thresh, y_pp, x_vals):
    pp_r = [np.floor(pp*100.) for pp in y_pp]
    pp_xs = {}
    for x_, p_ in zip(x_vals, pp_r):
        if p_ not in pp_xs.keys():
            pp_xs[p_] = x_

    sortedkeys = sorted(pp_xs.keys())
    if thresh < np.min(sortedkeys):
        thresh = np.min(sortedkeys)
    if thresh > np.max(sortedkeys):
        thresh = np.max(sortedkeys)

    return pp_xs, thresh


def calculate_rates(ones, zeros, pp_xs, thresh):
    rate_dict = {}
    rate_dict['tps'] = (ones > pp_xs[thresh])
    rate_dict['fps'] = (zeros > pp_xs[thresh])
    rate_dict['tns'] = (zeros < pp_xs[thresh])
    rate_dict['fns'] = (ones < pp_xs[thresh])

    rate_dict['tpr_crit'] = np.sum(rate_dict['tps'])/float(len(ones))
    rate_dict['fpr_crit'] = np.sum(rate_dict['fps'])/float(len(zeros))
    return rate_dict


def update(attrname, old, new):
    p = proportion.value
    s = separation.value
    t = threshold.value

    model_info = roc_data.calculate(p, s)
    x_vals = np.linspace(-1.,12.,300)
    y_pp = model_info['mod'].predict_proba(x_vals[:, np.newaxis])[:,1]


    ones_source.data = dict(x=model_info['ones'],
                            y=np.ones(len(model_info['ones'])))

    zeros_source.data = dict(x=model_info['zeros'],
                             y=np.zeros(len(model_info['zeros'])))

    lr_source.data =  dict(x=x_vals, y=y_pp)

    pp_xs, thresh = calculate_threshold(t, y_pp, x_vals)

    lr_thresh_source.data = dict(x0=[pp_xs[thresh], -1.],
                                 x1=[pp_xs[thresh], 12.0],
                                 y0=[0., thresh/100.],
                                 y1=[1.0, thresh/100.])

    roc_source.data = dict(x=model_info['fpr'],
                           y=model_info['tpr'])


roc_data = ROCData()

proportion = Slider(title="proportion", value=50, start=1, end=99, step=1)
separation = Slider(title="separation", value=50, start=1, end=99, step=1)
threshold = Slider(title='threshold', value=50, start=1, end=99, step=1)

model_info = roc_data.calculate(proportion.value, separation.value)

### Logreg plot:

x_vals = np.linspace(-1.,12.,300)
y_pp = model_info['mod'].predict_proba(x_vals[:, np.newaxis])[:,1]

ones_source = ColumnDataSource(data=dict(x=model_info['ones'],
                                         y=np.ones(len(model_info['ones']))))

zeros_source = ColumnDataSource(data=dict(x=model_info['zeros'],
                                          y=np.zeros(len(model_info['zeros']))))

lr_source = ColumnDataSource(data=dict(x=x_vals, y=y_pp))

lr_plot = figure(plot_height=350, plot_width=350, title="title",
                 toolbar_location=None, tools="")

lr_plot.circle(x="x", y="y", source=ones_source,
               size=10, line_color=None, color='#BCBD22')

lr_plot.circle(x="x", y="y", source=zeros_source,
               size=10, line_color=None, color='#9467BD')

lr_plot.line(x="x", y="y", source=lr_source, line_width=5, color='black')


pp_xs, thresh = calculate_threshold(threshold.value, y_pp, x_vals)

rate_dict = calculate_rates(model_info['ones'], model_info['zeros'],
                            pp_xs, thresh)

lr_thresh_source = ColumnDataSource(data=dict(x0=[pp_xs[thresh], -1.],
                                              x1=[pp_xs[thresh], 12.],
                                              y0=[0., thresh/100.],
                                              y1=[1.0, thresh/100.]))


lr_plot.segment(x0="x0", y0="y0", x1="x1", y1="y1",
                color='#7F7F7F', line_width=3,
                line_dash='dashed', source=lr_thresh_source)


### ROC Plot:
roc_plot = figure(plot_height=350, plot_width=350, title="ROC",
                  toolbar_location=None, tools="")

roc_plot.segment(x0=0., y0=0., x1=1., y1=1., color='#7F7F7F',
                 line_width=3, line_dash='dashed')

roc_source = ColumnDataSource(data=dict(x=model_info['fpr'],
                                        y=model_info['tpr']))

roc_plot.line(x='x', y='y', source=roc_source, line_width=3,
              color='#17BECF')


controls = [proportion, separation, threshold]
for control in controls:
    control.on_change('value', update)

inputs = widgetbox(*controls)

output_layout = layout([
    [inputs],
    [lr_plot, roc_plot]
], sizing_mode='scale_width')

curdoc().add_root(output_layout)
curdoc().title = "Confusion"
