# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from django.core.files.storage import default_storage

from django.core.paginator import Paginator, PageNotAnInteger, EmptyPage
from django.db.models.fields.files import FieldFile
from django.views.generic import FormView
from django.views.generic.base import TemplateView
from django.contrib import messages

from .forms import ContactForm, FilesForm, ContactFormSet

from bokeh.plotting import figure
from bokeh.resources import CDN
from bokeh.embed import components

from bokeh.layouts import row, column
from bokeh.models import BoxSelectTool, LassoSelectTool, Spacer
from bokeh.plotting import figure, curdoc

from bokeh.plotting import figure
from bokeh.models import Range1d
from bokeh.embed import components

import numpy as np


# http://yuji.wordpress.com/2013/01/30/django-form-field-in-initial-data-requires-a-fieldfile-instance/
class FakeField(object):
    storage = default_storage


fieldfile = FieldFile(None, FakeField, 'dummy.txt')


class HomePageView(TemplateView):
    template_name = 'main/home.html'


    def get_context_data(self, **kwargs):
        context = super(HomePageView, self).get_context_data(**kwargs)
        messages.info(self.request, 'hello http://example.com')
        
        # create some data
        x1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y1 = [0, 8, 2, 4, 6, 9, 5, 6, 25, 28, 4, 7]
        x2 = [2, 5, 7, 15, 18, 19, 25, 28, 9, 10, 4]
        y2 = [2, 4, 6, 9, 15, 18, 0, 8, 2, 25, 28]
        x3 = [0, 1, 0, 8, 2, 4, 6, 9, 7, 8, 9]
        y3 = [0, 8, 4, 6, 9, 15, 18, 19, 19, 25, 28]

        # select the tools we want
        TOOLS="pan,wheel_zoom,box_zoom,reset,save"

        # the red and blue graphs will share this data range
        xr1 = Range1d(start=0, end=30)
        yr1 = Range1d(start=0, end=30)

        # only the green will use this data range
        xr2 = Range1d(start=0, end=30)
        yr2 = Range1d(start=0, end=30)

        # build our figures
        p1 = figure(x_range=xr1, y_range=yr1, tools=TOOLS, plot_width=300, plot_height=300)
        p1.scatter(x1, y1, size=12, color="red", alpha=0.5)

        p2 = figure(x_range=xr1, y_range=yr1, tools=TOOLS, plot_width=300, plot_height=300)
        p2.scatter(x2, y2, size=12, color="blue", alpha=0.5)

        p3 = figure(x_range=xr2, y_range=yr2, tools=TOOLS, plot_width=300, plot_height=300)
        p3.scatter(x3, y3, size=12, color="green", alpha=0.5)

        # plots can be a single Bokeh Model, a list/tuple, or even a dictionary
        plot = {'Red': p1, 'Blue': p2, 'Green': p3}

        script, div = components(plot, CDN)

        context['bokeh_script'] = script
        context['figure'] = div

        return context


class DefaultFormsetView(FormView):
    template_name = 'main/formset.html'
    form_class = ContactFormSet

    # I think it's possible to move this to a single class that can handle this nav
    # active class crap
    def get_context_data(self, **kwargs):
        context = super(HomePageView, self).get_context_data(**kwargs)
        context['current_route'] = "formset"
        return context



class DefaultFormView(FormView):
    template_name = 'main/form.html'
    form_class = ContactForm


class DefaultFormByFieldView(FormView):
    template_name = 'main/form_by_field.html'
    form_class = ContactForm


class FormHorizontalView(FormView):
    template_name = 'main/form_horizontal.html'
    form_class = ContactForm


class FormInlineView(FormView):
    template_name = 'main/form_inline.html'
    form_class = ContactForm


class FormWithFilesView(FormView):
    template_name = 'main/form_with_files.html'
    form_class = FilesForm

    def get_context_data(self, **kwargs):
        context = super(FormWithFilesView, self).get_context_data(**kwargs)
        context['layout'] = self.request.GET.get('layout', 'vertical')
        return context

    def get_initial(self):
        return {
            'file4': fieldfile,
        }


class PaginationView(TemplateView):
    template_name = 'main/pagination.html'

    def get_context_data(self, **kwargs):
        context = super(PaginationView, self).get_context_data(**kwargs)
        lines = []
        for i in range(200):
            lines.append('Line %s' % (i + 1))
        paginator = Paginator(lines, 10)
        page = self.request.GET.get('page')
        try:
            show_lines = paginator.page(page)
        except PageNotAnInteger:
            # If page is not an integer, deliver first page.
            show_lines = paginator.page(1)
        except EmptyPage:
            # If page is out of range (e.g. 9999), deliver last page of results.
            show_lines = paginator.page(paginator.num_pages)
        context['lines'] = show_lines
        return context


class MiscView(TemplateView):
    template_name = 'main/misc.html'
