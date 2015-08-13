# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from pylab import *
from numpy import *
from scipy.ndimage import filters
import matplotlib.pylab as pyp
from PIL import Image

""" 

    Harris Corner detection

    Cian Conway - 10126767
    
""" 



al1 = np.array(Image.open('C:\\Users\\Cian\\Dropbox\\Final Semester\\Machine Vision\\Projects\\images\\Harris Corner\\cr1.png').convert("L")) #Local location on computer, replace with your own images
al2 = np.array(Image.open('C:\\Users\\Cian\\Dropbox\\Final Semester\\Machine Vision\\Projects\\images\\Harris Corner\\cr2.png').convert("L")) 

im_min, im_max = al1.min(), al1.max()

def harris(im,sigma=3): #Performs the Harris corner detector response function
    
    # derivatives
    imx = zeros(im.shape)
    filters.gaussian_filter(im, (sigma,sigma), (0,1), imx)
    imy = zeros(im.shape)
    filters.gaussian_filter(im, (sigma,sigma), (1,0), imy)
    
    # compute components of the Harris matrix
    Wxx = filters.gaussian_filter(imx*imx,sigma)
    Wxy = filters.gaussian_filter(imx*imy,sigma)
    Wyy = filters.gaussian_filter(imy*imy,sigma)
    
    # determinant and trace
    Wdet = Wxx*Wyy - Wxy**2
    Wtr = Wxx + Wyy
    
    return Wdet / Wtr
   
    
def harrisPoints(harrisim,min_dist=10,threshold=0.1): #Gets the Harris points and displays the corners. 
#min_dist is the minimum number of pixels seperating the corners and image boundary    
    
    # find top corner candidates
    corner_threshold = harrisim.max() * threshold
    harrisim_t = (harrisim > corner_threshold) * 1
    
    # get coordinates of candidates
    coords = array(harrisim_t.nonzero()).T
    
    #get values of candidates
    candidate_values = [harrisim[c[0],c[1]] for c in coords]
    
    # sort candidates
    index = argsort(candidate_values)[::-1]
    
    # store allowed point locations in array
    allowed_locations = zeros(harrisim.shape)
    allowed_locations[min_dist:-min_dist,min_dist:-min_dist] = 1
    
    # select the best points using min_distance
    filtered_coords = []
    for i in index:
        if allowed_locations[coords[i,0],coords[i,1]] == 1:
            filtered_coords.append(coords[i])
            allowed_locations[(coords[i,0]-min_dist):(coords[i,0]+min_dist),
                        (coords[i,1]-min_dist):(coords[i,1]+min_dist)] = 0
    
    return filtered_coords
    
    
def plot_harris_points(image,filtered_coords): #Plots Harris corners on graph
    
    figure()
    gray()
    imshow(image)
    plot([p[1] for p in filtered_coords],
                [p[0] for p in filtered_coords],'*')
    axis('off')
    show()
    

def get_descriptors(image,filtered_coords,wid=5): #return pixel value
    
    desc = []
    for coords in filtered_coords:
        patch = image[coords[0]-wid:coords[0]+wid+1,
                            coords[1]-wid:coords[1]+wid+1].flatten()
        desc.append(patch)
    
    return desc


def match(desc1,desc2,threshold=0.5): #Match method, for each corner in image 1, selects it's match in image 2 using normalized cross corelation
    
    n = len(desc1[0])
    
    # pair-wise distances
    d = -ones((len(desc1),len(desc2)))
    for i in range(len(desc1)):
        for j in range(len(desc2)):
            d1 = (desc1[i] - mean(desc1[i])) / std(desc1[i])
            d2 = (desc2[j] - mean(desc2[j])) / std(desc2[j])
            ncc_value = sum(d1 * d2) / (n-1)
            if ncc_value > threshold:
                d[i,j] = ncc_value
            
    ndx = argsort(-d)
    matchscores = ndx[:,0]
    
    return matchscores


def match_twosided(desc1,desc2,threshold=0.5): #Same as above, two sided symmetric version
    
    matches_12 = match(desc1,desc2,threshold)
    matches_21 = match(desc2,desc1,threshold)
    
    ndx_12 = where(matches_12 >= 0)[0]
    
    # remove matches that are not symmetric
    for n in ndx_12:
        if matches_21[matches_12[n]] != n:
            matches_12[n] = -1
    
    return matches_12


def appendimages(im1,im2): #the appended images displayed side by side for image mapping
    
    # select the image with the fewest rows and fill in enough empty rows
    rows1 = im1.shape[0]
    rows2 = im2.shape[0]
    
    
    #if elseif statement
    if rows1 < rows2:
        im1 = concatenate((im1,zeros((rows2-rows1,im1.shape[1]))),axis=0)
    elif rows1 > rows2:
        im2 = concatenate((im2,zeros((rows1-rows2,im2.shape[1]))),axis=0)
    
    
    return concatenate((im1,im2), axis=1)
    
    
def plot_matches(im1,im2,locs1,locs2,matchscores,show_below=True): #Plots the matches between both images
    
    im3 = appendimages(im1,im2)
    if show_below:
        im3 = vstack((im3,im3))
    
    imshow(im3)
    
    cols1 = im1.shape[1]
    for i,m in enumerate(matchscores):
        if m>0:
            plot([locs1[i][1],locs2[m][1]+cols1],[locs1[i][0],locs2[m][0]],'c')
    axis('off')
    
wid = 5
harrisim = harris(al1,3)
filtered_coords1 = harrisPoints(harrisim,wid+1)
d1 = get_descriptors(al1,filtered_coords1,wid)

harrisim = harris(al2,3)
filtered_coords2 = harrisPoints(harrisim,wid+1)
d2 = get_descriptors(al2,filtered_coords2,wid)

matches = match_twosided(d1,d2)

figure()
gray()
plot_matches(al1,al2,filtered_coords1,filtered_coords2,matches)
show()

# <codecell>


