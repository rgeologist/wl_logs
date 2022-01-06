# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 11:10:57 2021

@author: Raymi Castilla
"""
from itertools import cycle
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import pandas as pd
import numpy as np



class Wlogs_plot():
    
    def __init__(self, plot_obj, **kwargs):
        
        kwargs.update(constrained_layout=False)
        
        self.fig = plt.figure(**kwargs)
        
        if isinstance(plot_obj, pd.Series):
            self.log = plot_obj
        elif isinstance(plot_obj, pd.DataFrame):
            self.im_log = plot_obj
        elif isinstance(plot_obj, list):
            self.data_grid_list = plot_obj
        else:
            raise TypeError('plot_obj is not recognized as a valid argument')
        
        self.axes_grid=[]
    
    def __iter__(self):
        axes = self.flatten_axes_grid()
        for ax in axes:
            yield ax
    
    def plot(self, **kwargs):        
        if hasattr(self, 'log'):
            ax = plt.subplot(111)
            self._plot_data(self.log, ax, **kwargs)
            self.axes_grid = [ax]
        elif hasattr(self, 'im_log'):
            ax = plt.subplot(111)
            self._plot_data(self.im_log, ax, **kwargs)
            self.axes_grid = [ax]
        elif hasattr(self, 'data_grid_list'):
            self.axes_grid=self._axes_grid()
            params_nested_lst = kwargs.pop('plot_kws', self.make_nested_list(fill_value={}))
            
            outer_zip = zip(self.axes_grid, self.data_grid_list, params_nested_lst)
            for tracks_cluster, data_cluster, plot_params_lst in outer_zip:
                self._plot_in_cluster(tracks_cluster, data_cluster, plot_params_lst)
    
    
    def _plot_in_cluster(self, axes_list, data_list, params_list):
        first_axis = True
        
        iterator = zip(axes_list, data_list, params_list)
        for ax, data, params in iterator:
            self._plot_data(data, ax, **params)
            
            if first_axis:
                if not ax.yaxis_inverted():
                    ax.invert_yaxis()
            else:
                ax.yaxis.set_tick_params(length=0)
                ax.set(ylabel='')
            first_axis=False
        
    def set_size_cm(self, width, height):        
        self.fig.set_size_inches(width*0.393701, height*0.393701)        
    
    def set_size_inches(self, width, height):        
        self.fig.set_size_inches(width, height)
    
    @staticmethod
    def list_is_nested(a_list):
        is_nested=any([isinstance(lst, list) for lst in a_list])
        return is_nested
    
    def _outer_gridspec(self, **kwargs):        
        num_rows = 1
        num_cols = len(self.data_grid_list)        
        outer_grid = self.fig.add_gridspec(num_rows, num_cols, **kwargs)
        
        return outer_grid
    
    def _inner_gridspec(self, outer_gridspec_elem, num_cols, **kwargs):
        inner_gridspec = outer_gridspec_elem.subgridspec(1, num_cols, wspace=0, **kwargs)
        return inner_gridspec
    
    def _axes_grid(self, **kwargs):
        outer_grid = self._outer_gridspec()
        outer_lst = []
        for grdspec_elem, data_elem in zip(outer_grid, self.data_grid_list): 
            num_cols = len(data_elem)
            inner_grid = self._inner_gridspec(grdspec_elem, num_cols, **kwargs)
            axes = inner_grid.subplots(sharey=True)
            try:
                axes=list(axes)
            except TypeError:
                axes=[axes]
            outer_lst.append(axes)
        return outer_lst
    
    def _plot_data(self, data, ax, **kwargs):
        
        if isinstance(data, dict):
            func = data['func']
            f_args = data['args']
            f_kwargs = data['kwargs']
            loaded_data = func(*f_args, **f_kwargs)
            self._plot_data(loaded_data, ax, **kwargs)
        
        if isinstance(data, pd.Series):
            print('plot series')
            self._plot_curve(data, ax, **kwargs)
            
        if isinstance(data, pd.DataFrame):
            print('plot image')
            self._plot_image(data, ax, **kwargs)
        
        
    def _plot_curve(self, series, ax, **kwargs):
        ax.plot(series.values, series.index,  **kwargs)
        ax.grid()
    
    def _plot_image(self, dataframe, ax, *args, **kwargs):
        # breakpoint()
        kwargs.update(ax=ax)
        dataframe.bhim.plot_image(*args, **kwargs)
    
    @property
    def number_of_axes(self):        
        n_axes = len(self.flatten_axes_grid())
        return n_axes
    
    def flatten_axes_grid(self):  
        from matplotlib.cbook import flatten
        lst = list(flatten(self.axes_grid))
        return lst    
    
    @property
    def all_axes(self):
        return self.flatten_axes_grid()
    
    
    @property
    def _y_spans_data(self):
        yspans = [abs(ax.get_ylim()[0]-ax.get_ylim()[1]) for ax in self]
        return yspans
    
    @property
    def max_y_span_data(self):
        return max(self._y_spans_data)
    
    @property
    def min_y_span_data(self):
        return min(self._y_spans_data)
    
    @property
    def ax_max_y_span_data(self):
        index = self._y_spans_data.index(self.max_y_span_data)
        return self.flatten_axes_grid()[index]
        
    
    def set_tracks_width(self, auto=True, width=0.2,
                         left_pad=0.1, yaxis_pad=0.1, right_pad=0.05,
                         **kwargs):
        #width of tracks in fraction of figure width
        
        if not hasattr(self, 'axes_grid'):
            return
        
        if auto:
            width = self.compute_tracks_width_auto(left_pad=left_pad,
                                                   yaxis_pad=yaxis_pad,
                                                   right_pad=right_pad)
        
        first_cluster = True
        for tracks_cluster in self.axes_grid:
            first_track=True
            for track in tracks_cluster:
                track_bbox = track.get_position()
                if first_cluster and first_track:
                    x0 = left_pad
                    # x0 = track_bbox.xmin
                    first_track=False
                
                x1 = x0+width
                track_bbox.update_from_data_x([x0, x1], ignore=True)
                x0 = x1
                
                track.set_position(track_bbox)
            # return
            x0 += yaxis_pad
            first_cluster=False
            
    def compute_tracks_width_auto(self, left_pad=0.05, yaxis_pad=0.1, right_pad=0.05):        
        n_clusters = len(self.axes_grid)        
        n_axes     = self.number_of_axes
        width = (1-left_pad-right_pad-(n_clusters-1)*yaxis_pad)/n_axes
        return width
    
    def unify_y_scale(self, v_alignment='center'):
        #change scale and position of axes
        max_yinterval = self.max_y_span_data
        max_yinterval_track = self.ax_max_y_span_data
        highest_track_bbox = max_yinterval_track.get_position()
        max_height = highest_track_bbox.height
        resize_factor = max_height/max_yinterval
        # resize_factors = [yinterval/max_yinterval for yinterval in self._y_spans_data]
        # breakpoint()
        
        for ax in self:
            y_lims = ax.get_ybound()
            y_span = y_lims[1]-y_lims[0]
            new_height = y_span * resize_factor
            old_bbox = ax.get_position()
            # old_bounds = old_bbox.bounds
            height = new_height
            left   = old_bbox.x0
            width  = old_bbox.width
            
            if v_alignment == 'bottom':
                bottom = highest_track_bbox.y0
            elif v_alignment == 'center':
                bottom = highest_track_bbox.y0 + (max_height-height)/2
            elif v_alignment == 'top':
                bottom = highest_track_bbox.y0 + (max_height-height)       
            
            # height = old_bounds[3]*f
            # left   = old_bounds[0]
            # bottom = old_bounds[1] + (max_height-height)/2
            # width  = old_bounds[2]
            new_bbox = Bbox.from_bounds(left, bottom, width, height)
            ax.set_position(new_bbox) 
    

    def make_nested_list(self, nested_list=None, fill_value=''):
        if nested_list is None:
            nested_list = self.axes_grid
        new_lst=[]
        for elem in nested_list:
            if isinstance(elem, list):
                inner_list = self.make_nested_list(elem, fill_value)
                new_lst.append(inner_list)
            else:
                new_lst.append(fill_value)
        return new_lst
    
    #----Main
if __name__ == '__main__':
    
    pass

