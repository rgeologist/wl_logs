# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 10:55:13 2021

@author: Raymi Castilla
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import numpy as np
import progressbar as pgb
from scipy import ndimage
from scipy.interpolate import RectBivariateSpline
from skimage import exposure, filters
from zipfile import ZipFile, is_zipfile
from copy import deepcopy
import stecto as st

try:
    #delete the accesor to avoid warning from pandas
    del pd.DataFrame.imlog
except AttributeError:
    pass

@pd.api.extensions.register_dataframe_accessor('imlog')
class BHImage:
    
    def __init__(self, df, *args, **kwargs):     
        self.data = df
    
    @classmethod
    # @staticmethod
    def from_file(cls, fullpath, *args, **kwargs):
        
        filename      = kwargs.pop('filename', None)
        image_type    = kwargs.pop('image_type', 'single_array')        
        
        if is_zipfile(fullpath):
            df_image = cls._from_zip_file(fullpath, filename, image_type, **kwargs)
          
        else:            
            if image_type == 'rgb':
                df_image = cls._read_rgb_file(fullpath, **kwargs)                    
            elif image_type == 'single_array':
                df_image = cls._read_single_array_file(fullpath, **kwargs)        
        
        return df_image.copy()
    
    
    @classmethod
    def _from_zip_file(cls, fullpath, filename, image_type, **kwargs):
        
        with ZipFile(fullpath) as myzip:
            
            check_file_lst = [filename in item for item in myzip.namelist()]
            if any(check_file_lst):
                file_found_index = check_file_lst.index(True)
                log_file = myzip.namelist()[file_found_index]
            else:
                raise ValueError(f'Could not find log named {filename} inside zip file')
                        
            with myzip.open(log_file, 'r') as log:                
                if image_type == 'rgb':
                    df_image = cls._read_rgb_file(log, **kwargs)                    
                elif image_type == 'single_array':
                    df_image = cls._read_single_array_file(log, **kwargs)                    
        return df_image
    
    @staticmethod
    def _read_single_array_file(path, **kwargs):
        print('\nloading single array log from file!!!')
        
        start_md      = kwargs.pop('start_md', -np.inf)
        end_md        = kwargs.pop('end_md', np.inf)
        
        df_image = pd.read_csv(path, index_col=0, **kwargs)
        df_image.index.name = 'depth'
        #name columns
        num_cols = len(df_image.columns)
        df_image.columns=np.linspace(0, 360, num_cols)
        df_image = df_image.imlog.crop_depth(start_md=start_md, end_md=end_md)
        return df_image
    
    @staticmethod
    def _read_rgb_file(file, **kwargs):
        print('\nloading rgb log from file. This might take a while!!!')
        
        skiprows = kwargs.pop('skiprows', None)
        start_md = kwargs.pop('start_md', -np.inf)
        end_md   = kwargs.pop('end_md', np.inf)
        
        all_values_lst = []
        depth_lst=[]           
        for n, line in pgb.progressbar(enumerate(file)):
            line = line.decode()
            if n in [0, skiprows]:
                continue
            lst = line.split(',')
            depth_val = float(lst.pop(0))
            if depth_val<start_md:
                continue
            elif depth_val>end_md:
                break
            depth_lst.append(depth_val)
            line_lst=[]
            for rgb_trio in lst:
                line_lst.extend([int(v) for v in rgb_trio.strip().split('.')])
            all_values_lst.append(line_lst)            
        
        index_tuples = []
        n_cols = int(len(all_values_lst[0])/3)
        for angle in np.linspace(0,360,n_cols):
            for color in ['r','g','b']:
                index_tuples.append((angle, color))
                
        df = pd.DataFrame(all_values_lst, index=depth_lst,
                          columns=pd.MultiIndex.from_tuples(index_tuples))
        
        return df
    
    def crop_depth(self, start_md=None, end_md=None):
        
        if (start_md is None) and (end_md is None):
            return self.data
            
        if start_md is None:
            start_md = self.data.index.min()
        if end_md is None:
            end_md = self.data.index.max()
        
        resolution = self.data.index.to_series().diff().mean()
        new_index  = pd.Index(np.arange(start_md,end_md,resolution))
        df_image=self.data.reindex(index=new_index, method='nearest')
        
        return df_image
    
    def plot_image(self, *args, **kwargs):    
        
        df_image = self.data
        ax       = kwargs.pop('ax', None)
        colorbar = kwargs.pop('colorbar', False)
        
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        
        if df_image.columns.nlevels > 1:
            #the image is multichannel
            try:
                arr_to_plot = df_image.imlog.rgba_to_array()
                print('image is rgba')
            except KeyError:                
                #the image might be rgb
                try:
                    arr_to_plot = df_image.imlog.rgb_to_array()
                    print('image is rgb')
                except KeyError:  
                    raise KeyError('The image seems in multichannel mode but could not be transformed to an array')
                        
        elif df_image.columns.nlevels == 1:
            #the image is most certainly in single array mode
            arr_to_plot = df_image.values      
            print('image is single array')      
        
        log_extent = (0, 360, df_image.index.max(), df_image.index.min())
        cax = ax.imshow(arr_to_plot, extent=log_extent, aspect='auto', **kwargs)
            
        if not ax.yaxis_inverted():
            ax.invert_yaxis()                
            
        if colorbar:
            cbar = fig.colorbar(cax)
            
        ax.set_ylabel('MD (m)')
        ax.set_xlabel('degrees (°)')        
        ax.set_xticks(range(0,360,90))
        ax.grid()
        if colorbar:
            cbar.ax.set_ylabel('')
        
        return fig, ax
    
    def to_shaded_image(self, **kwargs):
        '''
        Uses matplotlibs's shade function to generate a shaded relief
        version of the log

        Parameters
        ----------
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        azdeg  = kwargs.pop('azdeg', 0)
        altdeg = kwargs.pop('altdeg', 20)
        vmin   = kwargs.pop('vmin', None)
        vmax   = kwargs.pop('vmax', None)
        cmap   = kwargs.pop('cmap', 'viridis')
        
        if vmin is None:
            vmin = np.nanpercentile(self.data.values, 5)
        if vmax is None:
            vmax = np.nanpercentile(self.data.values, 95)
        
        ls = LightSource(azdeg=azdeg,
                         altdeg=altdeg)
        z = ls.shade(self.data.values, plt.get_cmap(cmap),
                     vmin=vmin,
                     vmax=vmax, **kwargs)
        
        z = z.flatten().reshape((z.shape[0],-1))
        new_columns= pd.MultiIndex.from_product((self.data.columns, ['r','g','b','a']))
        
        df = pd.DataFrame(z, columns=new_columns, index=self.data.index)
        
        return df
    
    def to_binary(self, percentile_threshold=90):
        limit = np.nanpercentile(self.data.values, percentile_threshold)
        
        new_arr = np.empty_like(self.data.values)
        new_arr[self.data.values>=limit]=1
        new_arr[self.data.values<limit] =0
        return self.array_to_df(new_arr)
    
    def apply_median_filter(self, size=20):
        data = ndimage.median_filter(self.data.values, size=size)
        df = self.array_to_df(data)
        return df
        
    def equalize_adapthist(self, **kwargs):
        if any(self.data.dtypes == float):
            df = self.data.imlog.min_max_scaling()
        else:
            df = self.data
            
        image_eq = exposure.equalize_adapthist(df.values, **kwargs)
        df_eq = self.array_to_df(image_eq)
        return df_eq
    
    def equalize_histogram(self, **kwargs):
        image_eq = exposure.equalize_hist(self.data.values, **kwargs)
        df = self.array_to_df(image_eq)
        return df        
    
    def min_max_scaling(self):
        '''
        Scales the dataframe so values are between 0 - 1

        Returns
        -------
        pd.DataFrame
            DESCRIPTION.

        '''
        df_max = self.data.max().max()
        df_min = self.data.min().min()        
        df_scaled = self.data.applymap(lambda x: (x-df_min)/(df_max-df_min), na_action='ignore')        
        return df_scaled
    
    def sharpen(self, kernel=None):
        if kernel is None:
            kernel = np.array([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]])
        highpass = ndimage.convolve(self.data.values, kernel)
        return self.array_to_df(highpass)
    
    def unsharp_masking(self, kernel=None):
        if kernel is None:
            kernel = np.array([[1, 4, 6, 4, 1],
                               [4,  16, 24, 16, 4],
                               [6, 24, -476, 24, 6],
                               [4,  16, 24, 16, 4],
                               [1, 4, 6, 4, 1]])
        blurred = ndimage.convolve(self.data.values, kernel)
        sharp = self.data.values + blurred
        return self.array_to_df(sharp)
    
    def highpass_gaussian(self, sigma, **kwargs):
        lowpass = ndimage.gaussian_filter(self.data.values, sigma, **kwargs)
        gauss_highpass = self.data.values - lowpass
        return self.array_to_df(gauss_highpass)
    
    def difference(self, other):        
        other = other.reindex(index=self.data.index, method='nearest')
        return other - self.data        
        
    def flatten_with_fracture(self, fracture, *args, **kwargs):
        
        pivot_point_index = kwargs.pop('pivot_pt_index', 0)
        df_im = self.data
        df_fr = fracture.polyline
        
        #MD that will be used as a pivot point
        md_pivot = df_fr.md.iloc[pivot_point_index]
        
        left_index = df_im.index - md_pivot
        left_df    = pd.DataFrame(df_im.iloc[:,0].values, index=left_index)
        for im_col, polyline_md in zip(df_im.iloc[:,1:], df_fr.md.iloc[1:]):
            
            right_index = df_im.index - polyline_md        
            right_df = pd.DataFrame(df_im[im_col].values, index=right_index)
            left_df = pd.merge_asof(left_df.sort_index(), right_df,
                          left_index=True, right_index=True, direction='nearest')
            if im_col == 10:
                pass
        
        left_df.columns = df_im.columns
        im_flattened_by_polyline = left_df
        return BHImage(data=im_flattened_by_polyline)
    
    def extract_data_along_fracture(self, fracture):
        
        y = self.data.columns
        x = self.data.index
        z = self.data.values
        interp_func = RectBivariateSpline(x, y, z)
        xi = fracture.polyline.md
        yi = fracture.polyline.index
        zi = interp_func.ev(xi, yi)
        return zi
    
    def build_fracture_df(self, *args, **kwargs):
        
        path_out = kwargs.pop('path_out', None)
        
        fig, ax = self.plot_image(shade=False, **kwargs)
        
        coord_lst = fig.ginput(-1, timeout=0, mouse_pop=2, mouse_stop=3)
        df = pd.DataFrame(coord_lst, columns=['x', 'md'])
        if path_out is None:
            pass
        else:
            df.to_csv(path_out, index=False)

        return Fracture(polyline=df)
        
    def is_rgb_image(self):        
        return all(self.data.dtypes=='object')
    
    def rgb_to_array(self):        
        
        #suppose a uniform type 
        dtype = self.data.dtypes[0][0]
        
        image = np.empty((self.data.shape[0], int(self.data.shape[1]/3), 3),
                         dtype= dtype) 
        # breakpoint()
        image[:,:,0] = self.data.loc[:,(slice(None),'r')]
        image[:,:,1] = self.data.loc[:,(slice(None),'g')]
        image[:,:,2] = self.data.loc[:,(slice(None),'b')]
        return image
    
    
    def rgba_to_array(self): 
        
        #suppose a uniform type 
        dtype = self.data.dtypes[0][0]
        
        image = np.empty((self.data.shape[0], int(self.data.shape[1]/4), 4),
                         dtype= dtype) 
        # breakpoint()
        image[:,:,0] = self.data.loc[:,(slice(None),'r')]
        image[:,:,1] = self.data.loc[:,(slice(None),'g')]
        image[:,:,2] = self.data.loc[:,(slice(None),'b')]
        image[:,:,3] = self.data.loc[:,(slice(None),'a')]
        return image
    
    def array_to_df(self, array):
        df = pd.DataFrame(array, index=self.data.index, columns=self.data.columns)
        return df
    
class Fracture():
    
    def __init__(self, *args, **kwargs):
        
        #polyline is a DataFrame with the coordinates of the "sinusoidal"
        #representing the fracture in the borehole wall
        self.polyline = kwargs.pop('polyline', None)
        
        self.strike = kwargs.pop('strike', None)
        self.dip    = kwargs.pop('dip', None)
        
    def load_polyline(self, path_in, *args, **kwargs):
        
        df_stru = pd.read_csv(path_in, index_col='x')        
        self.polyline = df_stru
        

    def interpolate_structure(self, *args, **kwargs):
        
        precision = 5
        
        #Round index values to nearest 1/precision value.
        rnd_index = pd.Index(np.round(self.polyline.index.values*precision)/precision)
        
        self.polyline.index = rnd_index
        
        #Make sure there are no duplicates
        self.polyline = self.polyline.loc[~self.polyline.index.duplicated(),:]
        # self.polyline.plot(y='md', use_index=True, lw=0, marker='+',
        #                    color='tab:red', ax=ax)
        
        #Assign a new index to the original df
        new_index = pd.Index(np.round(np.arange(0, 360, 1/precision)*precision)/precision)
        self.polyline = self.polyline.reindex(index=new_index, copy=True)
        # self.polyline.plot(y='md', use_index=True, lw=0, marker='_',
        #                    color='tab:green', ax=ax)
        
        #Interpolate to fill the gaps
        self.polyline.interpolate(inplace=True, method='linear',
                                  limit_area=None, limit=100)
        self.polyline.fillna(method='bfill', inplace=True)
        self.polyline.fillna(method='ffill', inplace=True)
        
        #Keep only range(0,360)
        self.polyline = self.polyline.loc[range(360),:]
        
    def fracture_radius(self, cal_image):
        
        cal = cal_image.extract_data_along_fracture(self)
        poly      = self.polyline
        md        = poly.md.values
        md_center = (poly.md.max() + poly.md.min())/2
        
        radius = np.sqrt((md-md_center)**2+(cal)**2)
        
        self.polyline['radius'] = radius
    
    def offset_fracture(self, offset_mm):
        offset_m = offset_mm * 1e-3
        new_frac = deepcopy(self)
        
        new_frac.polyline.md = new_frac.polyline.md.apply(lambda x: x + offset_m)
        
        return new_frac
        
        
    def polar_plot_above_n_below(self, cal_image, offset_mm, **kwargs):
        
        frac_coords = kwargs.pop('frac_coords', False)
        borehole_az = kwargs.pop('borehole_az', None)
                
        fr_above = self.offset_fracture(-offset_mm)
        fr_below = self.offset_fracture(offset_mm)
        
        #fracture radius above and below
        fr_above.fracture_radius(cal_image)
        fr_below.fracture_radius(cal_image)
        if frac_coords:
            fr_above.borehole_to_fracture_coords(borehole_az)
            fr_below.borehole_to_fracture_coords(borehole_az)
        
        figp, axp = plt.subplots(subplot_kw={'projection':'polar'})
        axp.plot(np.radians(fr_above.polyline.index), fr_above.polyline.radius,
                  color='C2', label='Above Fr', ls='--')
        axp.plot(np.radians(fr_below.polyline.index), fr_below.polyline.radius,
                  color='C6', label='Below Fr')
        axp.set_theta_zero_location("N")
        axp.legend()
        return axp
        
    def borehole_to_fracture_coords(self, bh_azimuth):
        
        #Vertical plane parallel to borehole axis
        vert_pl = st.Plane(bh_azimuth, 90)
        
        frac_pl = st.Plane(self.strike, self.dip)
        
        #calculate the vector corresponding to the intersection of the
        # fracture plane and the vertical axial plane of the borehole
        hs_frac = vert_pl.intersection(frac_pl)
        
        dip_frac = frac_pl.dip_vector
        
        #rotation angle
        rotation_angle = round(hs_frac.angle_with(dip_frac))
        
        #sense of rotation        
        normal_vec = np.cross(dip_frac, hs_frac)
        if normal_vec[2] > 0:
            #rotation is clockwise
            rotation_angle *= -1
        
        #rotate
        index = self.polyline.index
        index_lst = list(index)
        index_lst = [el + rotation_angle for el in index_lst]
        #correct angles below 0°
        index_lst = [360+el if el<0 else el for el in index_lst]
        #correct angles above 360°
        index_lst = [el-360 if el>360 else el for el in index_lst]
                
        self.polyline.index = pd.Index(index_lst)
        
        return
    

#----MAIN

if __name__ == '__main__':
    
    ####### ATV
    path = r'test_data\ATV.zip'  
    fullpath = os.path.abspath(path)
    filename1 = 'Amplitude-HS.wax'
    start_md=177
    end_md=184
    atv = BHImage.from_file(path, filename=filename1, image_type='single_array',
                            start_md=start_md, end_md=end_md,
                            skiprows=1, na_values=-999)
    
    #interpolate
    atv = atv.interpolate()
    
    
    def versions_iter(atvlog):
        yield atvlog
        #sharpened version
        yield atvlog.imlog.sharpen()
        #binary version
        yield atvlog.imlog.to_binary(25)      
        #sobel filter
        yield atv.imlog.array_to_df(filters.sobel(atv.values))
        #relief
        yield atv.imlog.array_to_df(filters.sobel_h(atv.values))
    
    #plot        
    fig, axes = plt.subplots(1,5, sharey=True)
    fig.set_size_inches(9,10)
    
    def implot(atv, cmap, **kwargs):
        ax = kwargs.pop('ax', None)
        vmin = np.nanpercentile(atv.values, 10)
        vmax = np.nanpercentile(atv.values, 90)
        atv.imlog.plot_image(cmap=cmap, ax=ax,
                        vmin=vmin, vmax=vmax, **kwargs)
    
    cmaps = ['YlOrRd_r', 'YlOrRd_r', 'Greys_r',  'YlOrRd', 'Greys_r']
    
    for ax, atv_version, cmap in zip(axes, versions_iter(atv), cmaps):
        implot(atv_version, cmap, ax=ax)    
    
    #format plots
    titles = ['original', 'sharpened', 'b&w', 'edge detection\n(sobel)', 'relief\n(sobel_h)']
    for ax_ind, ax in enumerate(axes):
        if ax_ind>0:
            ax.set_ylabel('')
        ax.set_title(titles[ax_ind])
        xlabels= ax.get_xticklabels()
        for label in xlabels:
            label.set(fontsize=10)
    
    fig.tight_layout()
    
