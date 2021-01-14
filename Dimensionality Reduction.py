#!/usr/bin/env python
# -*- coding: utf-8 -*-
###########################
# CSCI 573 Data Mining - Dimensionality Reduction 
# Author: Chu-An Tsai
# 09/23/2019
###########################
import sys
import numpy as np

# read dataset 
script = sys.argv[0]
filename = sys.argv[1]
newdataset = np.loadtxt(filename, delimiter=",", usecols=(0,1,2,3,4,5,6,7,8,9))

########## Question a
##### calculate Z-score 
z_dataset = np.matrix(newdataset)
z_score = (z_dataset - z_dataset.mean(axis=0)) / (z_dataset.std(axis=0))
print("\nThe Z-normalization:")
print(z_score)

########## Question b
##### Compute the sample covariance matrix 
# Compute the mean vector
mean_vector = np.mean(z_score, axis=0)
# compute the centered data matrix
row_len = z_score.shape[0]
unit1 = np.matrix([1] * row_len)
unit1t = unit1.T
centered_data_matrix = z_score - (unit1t * mean_vector)
# compute the sample covariance matrix
sample_covariance_matrix = (centered_data_matrix.T.dot(centered_data_matrix)) / row_len
print("\nThe sample covariance matrix: ")
print(sample_covariance_matrix)

################### (Question b) answer - for testing
centered_data_matrix2 = np.array(centered_data_matrix.T)
test_samp_cov_dataset = np.cov(centered_data_matrix2, ddof=0)
print("\nThe sample covariance matrix(using np.cov):")
print(test_samp_cov_dataset)
###################

########## Question c
##### compute the dominant eigenvalue and eigenvector 
p_0 = np.matrix([1] * sample_covariance_matrix.shape[0]).T
p_1 = sample_covariance_matrix * p_0
p_2 = sample_covariance_matrix * (p_1 / np.max(p_1))
p_3 = sample_covariance_matrix * (p_2 / np.max(p_2))
p_4 = sample_covariance_matrix * (p_3 / np.max(p_3))
p_5 = sample_covariance_matrix * (p_4 / np.max(p_4))
p_6 = sample_covariance_matrix * (p_5 / np.max(p_5))
p_7 = sample_covariance_matrix * (p_6 / np.max(p_6))
p_8 = sample_covariance_matrix * (p_7 / np.max(p_7))
p_9 = sample_covariance_matrix * (p_8 / np.max(p_8))
p_10 = sample_covariance_matrix * (p_9 / np.max(p_9))
p_11 = sample_covariance_matrix * (p_10 / np.max(p_10))
p_12 = sample_covariance_matrix * (p_11 / np.max(p_11))
p_13 = sample_covariance_matrix * (p_12 / np.max(p_12))
p_14 = sample_covariance_matrix * (p_13 / np.max(p_13))
p_15 = sample_covariance_matrix * (p_14 / np.max(p_14))
p_16 = sample_covariance_matrix * (p_15 / np.max(p_15))
dominant_eigenvalue = np.max(p_16)/np.max((p_15 / np.max(p_15)))
norm_testing = np.sqrt(np.sum(np.square(p_16-p_15)))
# check if ||Xi-Xi-1|| < 0:000001 is true.
print("\nIs the norm less than 0.000001? ")
if (norm_testing < 0.000001):
    print("YES")
else:
    print("NO")
# normalize the eigenvector to unit length
dominant_eigenvector = p_16 / np.sqrt(np.sum(np.square(p_16)))
print("\nThe dominant eigrnvalue:")
print(dominant_eigenvalue)
print("\nThe dominant eigrnvector(normalized):")
print(dominant_eigenvector)

#################### (Question c) answer - for testing
eigenvalues_vector, eigenvectors_array = np.linalg.eig(sample_covariance_matrix) 
print("\nEigenvalues(using linalg.eig):")
print(eigenvalues_vector)
print("\nEigenvectors(using linalg.eig):")
print(eigenvectors_array)
##################

##########Question d
##### find two dominat eigenvalues and print the value of variance
# Sorting 
eigenvalues, eigenvectors = np.linalg.eig(sample_covariance_matrix) 
sorting = eigenvalues.argsort()[::-1]   
sort_eigenvalues = eigenvalues[sorting]
sort_eigenvectors = eigenvectors[:,sorting]

# first two dominat eigenvalues
first_two_dominat_eigenvectors = sort_eigenvectors[:, :2]
print("\nFirst two dominat eigenvectors: ")
print(first_two_dominat_eigenvectors)

# Projection of a data point onto first two dominant eigenvectors
projection_z = z_score.dot(first_two_dominat_eigenvectors)
z_score_t = z_score.T

# Variance of the data points in the projected subspace
two_dominat_eigen = sort_eigenvalues[0:2]    
variance = two_dominat_eigen.sum()        
print("\nThe value of the variance:")
print(variance)

########## Question e
##### print the covariance matrix in its eigen-decomposition form 
diagonal_array = np.diagflat(eigenvalues)
eigenvectors_T = eigenvectors.T
eigen_decom_form = eigenvectors.dot(diagonal_array.dot(eigenvectors_T))
print("\nThe covariance matrix in its eigen-decomposition form:")
print(eigen_decom_form)
print("\nThe eigrnvectors:")
print(eigenvectors)
print("\nDiagonalized eigenvalues:" )
print(diagonal_array)

########## Question f
##### write a subroutine to implement PCA Algorithm 

def subroutine_pca(D, set_up_percentage):
    # compute the centered data matrix
    mean_vector_pca = np.mean(D, axis=0)   
    row_len_pca = D.shape[0]
    unit1 = np.matrix([1] * row_len_pca)
    unit1t = unit1.T
    centered_data_matrix_pca = D - (unit1t * mean_vector_pca)
    # compute the sample covariance matrix
    sample_covariance_matrix_pca = (centered_data_matrix_pca.T.dot(centered_data_matrix_pca)) / row_len_pca
    # compute eigenvalues and eigenvectors and sorting
    eigvalues, eigvectors = np.linalg.eig(sample_covariance_matrix_pca)    
    sorting = eigvalues.argsort()[::-1]
    sorted_eigvalues = eigvalues[sorting]
    sorted_eigvectors = eigvectors[:,sorting]    
    total_variance = eigvalues.sum()
    number_of_eigenvalues = len(sorted_eigvalues)
    print("\nTotal number of eigenvalues: ")
    print(number_of_eigenvalues)
    # preserve 95% of variance
    count_eigvalues = 0
    for i in range(number_of_eigenvalues):
        find_percentage = [(k / total_variance) * 100 for k in sorted_eigvalues]
        real_percentage = np.cumsum(find_percentage)
        if (real_percentage[i] >= set_up_percentage):
            break
        count_eigvalues = count_eigvalues + 1
    new_count_eigvalues = count_eigvalues + 1          
    
    new_basis_vector = (sorted_eigvectors.T)[0:new_count_eigvalues,:]
    reduced_dataset = D.dot(new_basis_vector.T)
    # ten points tp be printed
    data_points_ten = reduced_dataset[0:10,:] 
    
    print("\nSelected eigenvector number:")
    print(new_count_eigvalues)
    
    return data_points_ten, sorted_eigvalues, reduced_dataset,  new_count_eigvalues 

D = z_score
x_percentage = 95

coordinate_data_points_ten, sorted_eigenvalues1 , reduced_matrix, counted_eigvalues = subroutine_pca(D,x_percentage)

print("\nThe coordinate of the first 10 data points:")
print(coordinate_data_points_ten)


########## Question g
##### compute the co-variance of the projected data points s
##### show that it matches with the sum of eigenvalues corresponding 
##### to principal vectors on which the data is projected.

covr_projected_data_points = np.cov(reduced_matrix, bias=True, rowvar=False)
print("\nThe covariance of the projected data points:")
print(covr_projected_data_points)
covariance_trace = np.trace(covr_projected_data_points)
print("\nTrace covariance:")
print(covariance_trace)
eigenvalue_sum = sorted_eigenvalues1[0:counted_eigvalues, ]
pricipal_vector_sum = eigenvalue_sum.sum()
print("\nSum of eigenvalues")
print(pricipal_vector_sum)

